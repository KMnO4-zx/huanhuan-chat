# ppo
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --model_name_or_path /root/autodl-tmp/ChatGLM2-6B/model \
    --stage ppo \
    --do_train \
    --use_v2 \
    --dataset zhenhuan \
    --finetuning_type lora \
    --resume_lora_training False \
    --checkpoint_dir ./output/sft \
    --reward_model ./output/rm \
    --output_dir ./output/ppo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --fp16