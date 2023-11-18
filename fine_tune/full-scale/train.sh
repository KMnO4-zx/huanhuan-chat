num_gpus=1

deepspeed --num_gpus $num_gpus train.py \
    --dataset_path ../../dataset/train/lora/huanhuan.json \
    --model_path /root/autodl-tmp/ChatGLM2-6B/model \
    --base_model ChatGLM2-6B \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 2400 \
    --save_steps 240 \
    --save_total_limit 10 \
    --learning_rate 1e-4 \
    --remove_unused_columns false \
    --logging_steps 10 \
    --output_dir ../../dataset/output \
    --deepspeed ds_config.json \
    --bf16 \