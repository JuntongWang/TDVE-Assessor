# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from functools import partial
from typing import List, Union

from datasets import Dataset as HfDataset

from swift.utils import (append_to_jsonl, get_logger, get_model_parameter_info, is_master, plot_images, stat_array,
                         use_torchacc)
from swift.llm.argument import TrainArguments
from swift.llm.base import SwiftPipeline
from swift.llm.dataset import EncodePreprocessor, GetLengthPreprocessor, PackingPreprocessor, load_dataset
from swift.llm.train.tuner import TunerMixin

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as Qwen2_5_VL_MLP
from utils.utils import LazyLLMDataset
from utils.template_base import Template as custom_template
from utils.qwen import Qwen2VLTemplate_Customized
from utils.trainer import Seq2SeqTrainer as CustomTrainer
import torch
import json
from scipy import stats
from tqdm import tqdm
import shutil


logger = get_logger()


class SwiftSft(SwiftPipeline, TunerMixin):
    args_class = TrainArguments
    args: args_class

    def __init__(self, args: Union[List[str], TrainArguments, None] = None) -> None:
        super().__init__(args)
        self.train_msg = {}
        self._prepare_model_tokenizer()
        self._prepare_template()

    def _prepare_model_tokenizer(self):
        args = self.args
        # Original model
        self.model, self.processor = args.get_model_processor()

        # Customize model
        self.model_MLP = Qwen2_5_VL_MLP.from_pretrained(
                args.self_model,
                torch_dtype=args.torch_dtype,
                device_map="auto"
        )
        self.model_MLP.mlp_score.to(self.model.device)

        attributes_to_copy = ['model_meta', 'embed_tokens', 'model_info', 'hf_device_map']
        for attr in attributes_to_copy:
            if hasattr(self.model, attr):
                setattr(self.model_MLP, attr, getattr(self.model, attr))

        if hasattr(self.model_MLP, 'hf_device_map'):
            logger.info(f'model.hf_device_map: {self.model.hf_device_map}')

        logger.info(f'model_info: {self.model_MLP.model_info}')

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
        if len(args.val_dataset) > 0:
            # Loading val dataset
            _, val_dataset = load_dataset(args.val_dataset, split_dataset_ratio=1.0, **dataset_kwargs)

        logger.info(f'val_dataset: {val_dataset}')

        return val_dataset


    def _get_data_collator(self):
        args = self.args
        template = self.template
        template_MLP = custom_template(self.processor, template.template_meta)
        padding_to = args.max_length if args.train_type == 'longlora' else None
        # return partial(template.data_collator, padding_to=padding_to)
        return partial(template_MLP.data_collator, padding_to=padding_to)

    def run(self):
        args = self.args
        val_dataset = self._get_dataset()
        val_dataset = self._encode_dataset(val_dataset)
        data_collator = self._get_data_collator()

        logger.info(f'model: {self.model_MLP}')
        del self.model
        torch.cuda.empty_cache()

        from torch.utils.data import DataLoader
        eval_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            collate_fn=data_collator,
        )

        pred_mos_list = []
        self.model_MLP.eval()
        for batch in tqdm(eval_dataloader):
            batch = {key: value.to(self.model_MLP.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            with torch.no_grad():
                outputs = self.model_MLP(**batch)
                pred_mos_list.append(float(outputs["score1"]))
                
        labels_qua = []
        with open("./testset.json", 'r') as f:
            datas = json.load(f)
        
            for data in datas:
                labels_qua.append(float(data['Authenticity']))
        
        print("------------------------------ MLP AUTHENTICITY ------------------------------")
        SRCC = stats.spearmanr(pred_mos_list, labels_qua)[0] 
        PLCC = stats.pearsonr(pred_mos_list, labels_qua)[0] 
        KRCC = stats.kendalltau(pred_mos_list, labels_qua)[0]
        print("SRCC: ", SRCC)
        print("PLCC: ", PLCC)
        print("KRCC: ", KRCC)
        
        shutil.rmtree("./output")

    def _stat_dataset(self, dataset: HfDataset):
        args = self.args
        dataset = GetLengthPreprocessor()(dataset, num_proc=args.dataset_num_proc)
        _, stat_str = stat_array(dataset['length'])
        logger.info(f'Dataset Token Length: {stat_str}')
        return stat_str

    def _encode_dataset(self, val_dataset):
        template = self.template
        args = self.args
        is_grpo = hasattr(args, 'rlhf_type') and args.rlhf_type == 'grpo'
        if not is_grpo:
            if args.lazy_tokenize:
                val_dataset = LazyLLMDataset(
                    val_dataset, template.encode, strict=args.strict, random_state=args.data_seed)
            else:
                preprocessor_cls = PackingPreprocessor if args.packing else EncodePreprocessor
                preprocessor = preprocessor_cls(template=template)
                val_dataset = preprocessor(val_dataset, num_proc=args.dataset_num_proc, strict=args.strict)

        return val_dataset


def sft_main(args: Union[List[str], TrainArguments, None] = None):
    return SwiftSft(args).main()


#----------------------------------------------------------------------

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Train the Qwen2.5-VL model with an MLP layer.")
    parser.add_argument('--model', type=str, required=True, help="Path to the pre-trained model or model identifier.")
    parser.add_argument('--val_dataset', type=str, required=True, help="Path to the validation dataset.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument("--self_model", type=str, required=True, help="Finetuned model path.")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train_args = TrainArguments(
        model=args.model,
        dataset=args.val_dataset.split(','),
        val_dataset=args.val_dataset.split(','),
        output_dir=args.output_dir,
    )
    train_args.self_model = args.self_model
    sft_main(train_args)
