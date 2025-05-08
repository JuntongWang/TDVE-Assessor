CUDA_VISIBLE_DEVICES=0 \
python eval_qua.py \
    --model '/home/wuyuyang/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct/' \
    --self_model '/home/wuyuyang/main/output/v3/final_model/' \
    --val_dataset '../../datasets/mlp/eval_quality.json' \
    --output_dir output \
