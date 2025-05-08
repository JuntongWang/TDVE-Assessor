# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from functools import partial
from typing import List, Union

from datasets import Dataset as HfDataset

from swift.plugin import extra_callbacks, get_loss_func, get_metric
from swift.trainers import IntervalStrategy, TrainerFactory
from swift.utils import (append_to_jsonl, get_logger, get_model_parameter_info, is_master, plot_images, stat_array,
                         use_torchacc)
from swift.llm.argument import TrainArguments
from swift.llm.base import SwiftPipeline
from swift.llm.dataset import EncodePreprocessor, GetLengthPreprocessor, PackingPreprocessor, load_dataset
from swift.llm.infer import prepare_generation_config
from swift.llm.model import get_model_arch
from swift.utils import deep_getattr
from swift.llm.utils import dynamic_gradient_checkpointing
from swift.llm.train.tuner import TunerMixin

from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as Qwen2_5_VL_MLP
from utils.utils import LazyLLMDataset
from utils.template_base import Template as custom_template
from utils.qwen import Qwen2VLTemplate_Customized
from utils.trainer import Seq2SeqTrainer as CustomTrainer
import torch
import gc


logger = get_logger()


class SwiftSft(SwiftPipeline, TunerMixin):
    args_class = TrainArguments
    args: args_class

    def __init__(self, args: Union[List[str], TrainArguments, None] = None) -> None:
        super().__init__(args)
        self.train_msg = {}
        self._prepare_model_tokenizer()
        self._prepare_template()
        self._prepare_callbacks()
        self.args.save_args()

    def _prepare_gradient_checkpointing(self):
        args = self.args

        if args.gradient_checkpointing:
            self.model_MLP.supports_gradient_checkpointing = True
            dynamic_gradient_checkpointing(self.model_MLP)
            self.model_MLP.config.use_cache = False  # fix transformers==4.36
            self.model_MLP.enable_input_require_grads()
        model_meta = self.model_MLP.model_meta
        model_arch = get_model_arch(model_meta.model_arch)
        if model_meta.is_multimodal and model_arch:
            for vision_tower_name in model_arch.vision_tower:
                vision_tower = deep_getattr(self.model, vision_tower_name)
                if hasattr(vision_tower, 'enable_input_require_grads'):
                    try:
                        vision_tower.enable_input_require_grads()
                    except NotImplementedError:
                        pass

    def _prepare_generation_config(self):
        args = self.args
        self.model.generation_config = prepare_generation_config(self.model.generation_config,
                                                                 args.get_request_config(), self.tokenizer)
        logger.info(f'model.generation_config: {self.model.generation_config}')
        self.model_MLP.generation_config = self.model.generation_config

    def _prepare_model_tokenizer(self):
        args = self.args
        # Original model
        self.model, self.processor = args.get_model_processor()

        # Customize model
        self.model_MLP = Qwen2_5_VL_MLP.from_pretrained(
                args.model,
                torch_dtype=args.torch_dtype
                #device_map="auto"
        )
        self.model_MLP.mlp_score.to(self.model.device)
        # Freeze all parameters first
        for name, param in self.model_MLP.named_parameters():
             param.requires_grad = False # Freeze everything

        # Then, unfreeze the parameters you want to train (mlp_score)
        for name, param in self.model_MLP.named_parameters():
            if "mlp_score" in name.lower():
                param.requires_grad = True
                logger.info(f"Setting {name} to requires_grad=True") # Optional: Log which params are trainable

        # Check if freeze_vit is True in args and apply it
        if args.freeze_vit:
            vision_tower_name = "vision_tower" # Common name, check your model's structure
            if hasattr(self.model_MLP, vision_tower_name):
                 vision_tower = getattr(self.model_MLP, vision_tower_name)
                 for name, param in vision_tower.named_parameters():
                     param.requires_grad = False
                     # logger.info(f"Freezing vision tower parameter: {name}") # Optional logging

        attributes_to_copy = ['model_meta', 'embed_tokens', 'model_info', 'hf_device_map']
        for attr in attributes_to_copy:
            if hasattr(self.model, attr):
                setattr(self.model_MLP, attr, getattr(self.model, attr))

        if hasattr(self.model_MLP, 'hf_device_map'):
            logger.info(f'model.hf_device_map: {self.model.hf_device_map}')

        logger.info(f'model_info: {self.model_MLP.model_info}')

        self._prepare_generation_config()
        self._prepare_gradient_checkpointing()

    def _prepare_template(self) -> None:
        template = self.args.get_template(self.processor)
        if self.args.task_type == 'causal_lm':
            template.set_mode('train')
        if template.use_model:
            template.model = self.model_MLP

        self.my_template = Qwen2VLTemplate_Customized(self.processor, template.template_meta)
        self.my_template.set_mode('train')
        self.my_template.model = self.model_MLP
        self.template = template

    def _get_dataset(self):
        # The random shuffling of the training set occurs in the dataloader of the trainer.
        args = self.args
        dataset_kwargs = args.get_dataset_kwargs()
        train_dataset, val_dataset = load_dataset(
            args.dataset, split_dataset_ratio=args.split_dataset_ratio, **dataset_kwargs)
        if len(args.val_dataset) > 0:
            # Loading val dataset
            _, val_dataset = load_dataset(args.val_dataset, split_dataset_ratio=1.0, **dataset_kwargs)
            assert args.split_dataset_ratio == 0.
        logger.info(f'train_dataset: {train_dataset}')
        logger.info(f'val_dataset: {val_dataset}')

        return train_dataset, val_dataset

    def _get_loss_func(self):
        args = self.args
        loss_type = args.loss_type
        if loss_type is None and args.loss_scale != 'default':
            loss_type = 'loss_scale'
        return get_loss_func(loss_type)

    def _get_data_collator(self):
        args = self.args
        template = self.template
        template_MLP = custom_template(self.processor, template.template_meta)
        padding_to = args.max_length if args.train_type == 'longlora' else None
        # return partial(template.data_collator, padding_to=padding_to)
        return partial(template_MLP.data_collator, padding_to=padding_to)

    def run(self):
        args = self.args

        train_dataset, val_dataset = self._get_dataset()
        if args.task_type == 'seq_cls' and isinstance(train_dataset, HfDataset) and 'label' in train_dataset.features:
            min_num_labels = int(max(train_dataset['label']) + 1)
            assert args.num_labels >= min_num_labels, (
                f'args.num_labels: {args.num_labels}, min_num_labels: {min_num_labels}')

        train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)
        data_collator = self._get_data_collator()

        # Some tuners require train_dataset and data_collator for preparation: LoRA-GA
        self.model_MLP = self.prepare_model(self.args, self.model_MLP, template=self.template, train_dataset=train_dataset)
        model_parameter_info = get_model_parameter_info(self.model_MLP)
        self.train_msg['model_parameter_info'] = model_parameter_info

        logger.info(f'model_parameter_info: {model_parameter_info}')
        logger.info(f'model: {self.model_MLP}')

        for param in self.model_MLP.mlp_score.parameters():
            param.requires_grad = True

        # trainer_cls = TrainerFactory.get_trainer_cls(args)
        trainer = CustomTrainer(
            model=self.model_MLP,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.my_template,
            **self._get_trainer_kwargs(),
        )

        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        return self.train(trainer)

    def _get_trainer_kwargs(self):
        args = self.args
        if args.metric is not None:
            compute_metrics, preprocess_logits_for_metrics = get_metric(args.metric)
        elif args.predict_with_generate:
            compute_metrics, preprocess_logits_for_metrics = get_metric('nlg')
        else:
            compute_metrics, preprocess_logits_for_metrics = get_metric('acc')
            compute_metrics = partial(
                compute_metrics,
                acc_strategy=args.acc_strategy,
                is_encoder_decoder=self.model.config.is_encoder_decoder)
        return {
            'compute_metrics': compute_metrics,
            'preprocess_logits_for_metrics': preprocess_logits_for_metrics,
            'compute_loss_func': self._get_loss_func()
        }

    def _save_trainer_state(self, trainer):
        training_args = trainer.args
        state = trainer.state

        if self.args.create_checkpoint_symlink:
            last_checkpoint = os.path.join(self.args.output_dir, 'last')
            best_checkpoint = os.path.join(self.args.output_dir, 'best')
            os.symlink(state.last_model_checkpoint, last_checkpoint)
            os.symlink(state.best_model_checkpoint, best_checkpoint)
            state.last_model_checkpoint = last_checkpoint
            state.best_model_checkpoint = best_checkpoint
        logger.info(f'last_model_checkpoint: {state.last_model_checkpoint}')
        logger.info(f'best_model_checkpoint: {state.best_model_checkpoint}')

        # Visualization
        if is_master() and not use_torchacc():
            if 'tensorboard' in training_args.report_to:
                images_dir = os.path.join(training_args.output_dir, 'images')
                logger.info(f'images_dir: {images_dir}')
                plot_images(images_dir, training_args.logging_dir, ['train/loss'], 0.9)
            if training_args.push_to_hub:
                trainer.push_to_hub()

        self.train_msg.update({
            'last_model_checkpoint': state.last_model_checkpoint,
            'best_model_checkpoint': state.best_model_checkpoint,
            'best_metric': state.best_metric,
            'global_step': state.global_step,
            'log_history': state.log_history,
            'memory': trainer.max_memory,
        })
        if is_master():
            jsonl_path = os.path.join(training_args.output_dir, 'logging.jsonl')
            append_to_jsonl(jsonl_path, self.train_msg)
        return self.train_msg

    def train(self, trainer):
        logging_path = os.path.join(trainer.args.output_dir, 'logging.jsonl')
        logger.info(f'The logging file will be saved in: {logging_path}')
        trainer.train(trainer.args.resume_from_checkpoint)
        trainer.save_eval_loss_image(os.path.join(trainer.args.output_dir, 'images'))

        mlp_state_dict = {}
        for name, param in self.model_MLP.named_parameters():
            if "mlp_score" in name.lower() and param.requires_grad:
                clean_name = name.replace("base_model.model.", "")
                mlp_state_dict[clean_name] = param.cpu().clone()

        if mlp_state_dict:
            logger.info(f'Saving mlp_score weights.')
            from safetensors.torch import save_file
            save_file(mlp_state_dict, os.path.join(trainer.args.output_dir, "mlp_layers.safetensors"))

        return self._save_trainer_state(trainer)

    def _prepare_callbacks(self):
        from swift.llm.train.callback import DynamicLayerActivationCallback, TrainerAdapterCallback
        args = self.args
        callbacks = []
        if args.lisa_activated_layers > 0:
            assert args.train_type == 'full', 'LISA only supports full parameter training.'
            lisa_callback = DynamicLayerActivationCallback(
                n_layers=args.lisa_activated_layers,  # Number of layers to activate
                step_interval=args.lisa_step_interval,  # Step interval to update active layers
                model=self.model_MLP)
            lisa_callback.switch_active_layers()  # Make trainable parameters printing a correct value
            callbacks.append(lisa_callback)

        if args.is_adapter and args.train_type == 'adalora':
            callbacks.append(TrainerAdapterCallback(args))
        callbacks += extra_callbacks
        self.callbacks = callbacks

    def _stat_dataset(self, dataset: HfDataset):
        args = self.args
        dataset = GetLengthPreprocessor()(dataset, num_proc=args.dataset_num_proc)
        _, stat_str = stat_array(dataset['length'])
        logger.info(f'Dataset Token Length: {stat_str}')
        return stat_str

    def _encode_dataset(self, train_dataset, val_dataset):
        template = self.template
        args = self.args
        is_grpo = hasattr(args, 'rlhf_type') and args.rlhf_type == 'grpo'
        if not is_grpo:
            if args.lazy_tokenize:
                train_dataset = LazyLLMDataset(
                    train_dataset, template.encode, strict=args.strict, random_state=args.data_seed)
                if val_dataset is not None and not args.predict_with_generate:
                    val_dataset = LazyLLMDataset(
                        val_dataset, template.encode, strict=args.strict, random_state=args.data_seed)
            else:
                preprocessor_cls = PackingPreprocessor if args.packing else EncodePreprocessor
                preprocessor = preprocessor_cls(template=template)
                train_dataset = preprocessor(train_dataset, num_proc=args.dataset_num_proc, strict=args.strict)
                if val_dataset is not None and not args.predict_with_generate:
                    val_dataset = preprocessor(val_dataset, num_proc=args.dataset_num_proc, strict=args.strict)

            inputs = train_dataset[0] if hasattr(train_dataset, '__len__') else next(iter(train_dataset))
            template.print_inputs(inputs, tokenizer_kwargs=inputs.pop('tokenizer_kwargs', None) or {})
            if isinstance(train_dataset, HfDataset):
                self.train_msg['train_dataset'] = self._stat_dataset(train_dataset)
                if val_dataset is not None and not args.predict_with_generate:
                    self.train_msg['val_dataset'] = self._stat_dataset(val_dataset)

        if val_dataset is None:
            args.training_args.evaluation_strategy = IntervalStrategy.NO
            args.training_args.eval_strategy = IntervalStrategy.NO
        return train_dataset, val_dataset


def sft_main(args: Union[List[str], TrainArguments, None] = None):
    return SwiftSft(args).main()


#----------------------------------------------------------------------

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Train the Qwen2.5-VL model with an MLP layer.")
    parser.add_argument('--model', type=str, required=True, help="Path to the pre-trained model or model identifier.")
    parser.add_argument('--train_type', type=str, required=True, help="Type of training (e.g., lora).")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the training dataset.")
    parser.add_argument('--val_dataset', type=str, required=True, help="Path to the validation dataset.")
    parser.add_argument('--split_dataset_ratio', type=float, default=0, help="Ratio to split the dataset.")
    parser.add_argument('--torch_dtype', type=str, default='bfloat16', help="Torch data type.")
    parser.add_argument('--freeze_llm', action='store_true', help="Whether to freeze the language model.")
    parser.add_argument('--freeze_vit', type=bool, default=True, help="Whether to freeze the vision transformer.")
    parser.add_argument('--num_train_epochs', type=int, default=1, help="Number of training epochs.")
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help="Batch size for training.")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--lora_rank', type=int, default=8, help="LoRA rank.")
    parser.add_argument('--lora_alpha', type=int, default=32, help="LoRA alpha.")
    parser.add_argument('--target_modules', type=str, default='all-linear', help="Target modules for LoRA.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument('--eval_steps', type=int, default=50, help="Evaluate every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=3200, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=5, help="Limit the total amount of checkpoints.")
    parser.add_argument('--logging_steps', type=int, default=5, help="Log every X updates steps.")
    parser.add_argument('--max_length', type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument('--system', type=str, default='You are a helpful assistant.', help="System prompt.")
    parser.add_argument('--weight_decay', type=float, default=0.1, help="Weight decay.")
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument('--dataloader_num_workers', type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument('--model_author', type=str, default='swift', help="Author of the model.")
    parser.add_argument('--model_name', type=str, default='swift-robot', help="Name of the model.")
    # parser.add_argument('--model_kwargs', type=str, help="Additional model kwargs.")
    # parser.add_argument('--freeze_aligner', type=bool, default=False, help="Whether to freeze the aligner.")
    return parser.parse_args()

os.environ['TORCH_DISTRIBUTED_BACKEND'] = 'nccl'
if __name__ == '__main__':
    args = parse_args()
    train_args = TrainArguments(
        model=args.model,
        train_type=args.train_type,
        dataset=args.dataset.split(','),
        val_dataset=args.val_dataset.split(','),
        split_dataset_ratio=args.split_dataset_ratio,
        torch_dtype=args.torch_dtype,
        freeze_llm=True,
        freeze_vit=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules.split(','),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        max_length=args.max_length,
        output_dir=args.output_dir,
        system=args.system,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=args.dataloader_num_workers,
        model_author=args.model_author,
        model_name=args.model_name,
        ddp_find_unused_parameters=True,
    )
    train_args.train_type = args.train_type
    train_args.freeze_llm = True
    train_args.freeze_vit = True
    sft_main(train_args)
