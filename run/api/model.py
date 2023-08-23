#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2023/08/23 17:23:14
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   初始化部署微调模型，供 API 调用
'''

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel

def init_GLM(model_path : str, lora_path : str, logger):
    # 初始化 ChatGLM 模型
    # 加载底座模型
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map='auto')
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("从{}加载模型成功".format(model_path))
    # 加载微调模型
    model = PeftModel.from_pretrained(model, lora_path)
    logger.info("加载 LoRa 参数成功")
    
    return model, tokenizer

def get_glm_completion(model, tokenizer, prompt, temprature=0, max_length=4096):
    with torch.no_grad():
        ids = tokenizer.encode(prompt)
        input_ids = torch.LongTensor([ids])
        out = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=False,
            temperature=temprature
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(prompt, "").replace("\nEND", "").strip()
        return answer

def init_BaiChuan(model_path : str, lora_path : str, logger):
    # 初始化 BaiChuan 模型
    # 加载底座模型
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("从{}加载模型成功".format(model_path))
    # 加载微调模型
    model = PeftModel.from_pretrained(model, lora_path).half()
    logger.info("加载 LoRa 参数成功")

    return model, tokenizer

def get_baichuan_completion(model, tokenizer, prompt, temprature=0, max_length=4096):
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,
            temperature=temprature
        )
        answer = tokenizer.decode(out[0][len(inputs.input_ids[0]):])
        return answer


