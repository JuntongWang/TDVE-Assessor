CUDA_VISIBLE_DEVICES=0 \
python eval_auth.py \
    --model '/home/wuyuyang/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct/' \
    --self_model '/home/wuyuyang/main/output/v3/final_model/' \
    --val_dataset '../../datasets/mlp/eval_authenticity.json' \
    --output_dir output \
