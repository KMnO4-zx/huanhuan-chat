# sft

CUDA_VISIBLE_DEVICES=0 python src/train_baichuan.py \
    --lora_rank 8 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --max_steps 600 \
    --save_steps 60 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 10 \
    --output_dir output/baichuan-sft