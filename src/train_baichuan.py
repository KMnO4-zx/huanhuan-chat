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
import bitsandbytes as bnb

MODEL_PATH = "baichuan-inc/Baichuan-7B"
DATASET = "/data/zhenhuan.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

@dataclass
class FinetuneArguments:
    dataset_path: str = field(default="data/alpaca")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["instruction"]
    target = example["output"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def read_jsonl(path, max_seq_length, skip_overlength=False):
    model_name = MODEL_PATH
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        lst = json.load(f)
        for example in tqdm(lst):
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        # print(ids)
        # print(feature)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

'''该函数在Float16精度下还有bug，暂时采用了Int8精度的网络层代替'''
def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    print(model.named_modules)
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            # print(names)
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    elif "model" in lora_module_names:
        lora_module_names.remove("model")
    # print(lora_module_names)
    return list(lora_module_names)



class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # print(model(input_ids=inputs["input_ids"],
        #     labels=inputs["labels"],
        # ))
        # 13B
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        )[0]
        # 7B
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def main():

    dataset_path = DATASET
    max_seq_length = 256
    skip_overlength = False


    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # init model
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                                trust_remote_code=True,
                                                # quantization_config=BitsAndBytesConfig(
                                                #     load_in_4bit=True,
                                                #     bnb_4bit_compute_dtype=torch.bfloat16,
                                                #     bnb_4bit_use_double_quant=True,
                                                #     bnb_4bit_quant_type='nf4'
                                                # ),
                                                device_map="auto")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    # model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    # target_modules = find_all_linear_names(model)
    target_modules = ['W_pack', 'down_proj', 'o_proj', 'gate_proj', 'up_proj']
    print(target_modules)
    # setup peft


# modules = find_all_linear_names(model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules= target_modules
    )
    model = get_peft_model(model, peft_config)
    # model = PeftModel.from_pretrained(model, "/root/autodl-tmp/output/lora-baichuan-continue-fifthdata-15k-8-1e-4/", is_trainable=True)

    # load dataset
    dataset = datasets.Dataset.from_generator(
            lambda: read_jsonl(dataset_path, max_seq_length, skip_overlength)
        )    
    print(f"\n{len(dataset)=}\n")

    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        # callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )
    trainer.train()
    # writer.close()
    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
