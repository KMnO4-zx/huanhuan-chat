# from transformers.integrations import TensorBoardCallback
# from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from dataclasses import dataclass, field
from transformers import BitsAndBytesConfig
import datasets
import os
import transformers
from tqdm import tqdm
import json
import sys
# 导入 log 模块目录
sys.path.append("../../")
from log.logutli import Logger

# dataclass：Python 类修饰符，数据类，封装了__init__()、 __repr__()和__eq__()函数
@dataclass
class FinetuneArguments:
    # 微调参数
    # field：dataclass 函数，用于指定变量初始化
    base_model: str = field(default="ChatGLM2")
    dataset_path: str = field(default="../../dataset/train/lora/huanhuan.json")
    model_path: str = field(default="../../dataset/model")
    lora_rank: int = field(default=8)
    max_seq_length: int = field(default=256)
    skip_overlength: bool = field(default=False)
    continue_training: bool = field(default=False)
    checkpoint: str = field(default=None)
    log_name: str = field(default="log")


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["instruction"]
    target = example["output"]
    full_prompt = prompt + target
    result = tokenizer(full_prompt, truncation=True,max_length=max_seq_length,padding=False,return_tensors=None)
    result["labels"] = result["input_ids"].copy()
    tokenized_user_prompt = tokenizer(prompt, truncation=True,max_length=max_seq_length,padding=False,return_tensors=None)
    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    result["labels"] = [
        -100
    ] * user_prompt_len + result["labels"][
        user_prompt_len:
    ] 
    return result


def read_jsonl(path, max_seq_length, model_path, logger, skip_overlength=False):
    model_name = model_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        lst = json.load(f)
        logger.debug("加载jsonl数据集，数据总量为{}".format(len(lst)))
        for example in tqdm(lst):
            feature = preprocess(tokenizer, config, example, max_seq_length)
            yield feature

class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 7B
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

class ModifiedTrainer13B(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 13B
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        )[0]

def main():

    # Parse 命令行参数
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    log_id     = 'train'  
    log_dir    = f'../../log/'
    log_name   = '{}.log'.format(finetune_args.log_name)
    log_level  = 'info'

    # 初始化日志
    logger = Logger(log_id, log_dir, log_name, log_level).logger
    logger.info('开始 finetune 训练')
    
    logger.debug("命令行参数")
    logger.debug("finetune_args:")
    logger.debug(finetune_args.__repr__())
    logger.debug("training_args:")
    logger.debug(training_args.__repr__())

    # 初始化底座模型
    
    tokenizer = AutoTokenizer.from_pretrained(finetune_args.model_path, trust_remote_code=True)
    if "ChatGLM" not in finetune_args.base_model:
        tokenizer.pad_token = tokenizer.eos_token

    # GLM 为底座
    if "ChatGLM" in finetune_args.base_model:
        model = AutoModel.from_pretrained(
            finetune_args.model_path,             
            trust_remote_code=True,
            from_tf=bool(".ckpt" in finetune_args.model_path),
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)}
        )
        logger.info("从{}加载模型成功".format(finetune_args.model_path))
    elif "BaiChuan" in finetune_args.base_model or "LLaMA"  in finetune_args.base_model:
        model = AutoModelForCausalLM.from_pretrained(
                finetune_args.model_path,
                trust_remote_code=True,
                from_tf=bool(".ckpt" in finetune_args.model_path),
                torch_dtype=torch.float16,
                device_map={"": int(os.environ.get("LOCAL_RANK") or 0)}
            )
        logger.info("从{}加载模型成功".format(finetune_args.model_path))
    else:
        logger.error("错误参数：底座模型必须是 ChatGLM、BaiChuan 或者 LLaMA-Chinese")
        raise ValueError("错误参数：底座模型必须是 ChatGLM、BaiChuan 或者 LLaMA-Chinese")

    logger.info("从{}加载模型成功".format(finetune_args.model_path))

    # 不一定需要
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # 加载数据集
    try:
        dataset = datasets.Dataset.from_generator(
                lambda: read_jsonl(finetune_args.dataset_path, finetune_args.max_seq_length, finetune_args.model_path, logger, finetune_args.skip_overlength)
            ) 
    except Exception as e:
        logger.error("从{}加载数据集失败".format(finetune_args.dataset_path))
        logger.error("错误信息为：")
        logger.error(e.__repr__())
        raise e   
    logger.info("从{}加载数据集成功".format(finetune_args.dataset_path))

    if "ChatGLM" in finetune_args.base_model:
        trainer = ModifiedTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
    elif "BaiChuan" in finetune_args.base_model  or "LLaMA"  in finetune_args.base_model:
        if finetune_args.base_model == "BaiChuan-13B":
            trainer = ModifiedTrainer13B(
                model=model,
                args=training_args,
                train_dataset=dataset,
                eval_dataset=None,
                tokenizer=tokenizer,
                # Data collator will default to DataCollatorWithPadding, so we change it.
                data_collator=transformers.DataCollatorForSeq2Seq(
                    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                ),
            )
        else:
            trainer = ModifiedTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                eval_dataset=None,
                tokenizer=tokenizer,
                # Data collator will default to DataCollatorWithPadding, so we change it.
                data_collator=transformers.DataCollatorForSeq2Seq(
                    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                ),
            )
    
    logger.info("成功加载 Trainer")

    train_result = trainer.train()
    trainer.save_model(training_args.output_dir)  # Saves the tokenizer too for easy upload
    trainer.save_state()
    logger.info("训练完成，训练结果保存在{}".format(training_args.output_dir))



if __name__ == "__main__":
    main()