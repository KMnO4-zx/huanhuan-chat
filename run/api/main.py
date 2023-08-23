#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/08/23 17:20:38
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   将微调后的模型封装为 API 服务
'''
from fastapi import FastAPI
from pydantic import BaseModel
from dataclasses import field
import sys
# 导入 log 模块目录
sys.path.append("../../")
from log.logutli import Logger

from model import *

# 路径参数
# 使用模型类型
BASE_MODEL = "ChatGLM2"
# 本地底座模型参数路径
MODEL_PATH = "../../dataset/model"
# 微调参数路径
LoRA_PATH = "../../dataset/output"

# 日志设置
log_id     = 'run_api'  
log_dir    = f'../../log/'
log_name   = 'log.log'
log_level  = 'debug'

# 初始化日志
logger = Logger(log_id, log_dir, log_name, log_level).logger
logger.info('部署API')

logger.debug("部署参数")
logger.debug("run_args:")
logger.debug("基座模型：" + BASE_MODEL)
logger.debug("基座模型参数：" + MODEL_PATH)
logger.debug("微调模型参数：" + LoRA_PATH)

app = FastAPI() # 创建 api 对象

# 定义一个数据模型，用于接收POST请求中的数据
class Item(BaseModel):
    prompt : str
    temperature : float  = field(default = 0.0)
    max_length : int = field(default = 4096)

if "BaiChuan" in BASE_MODEL:
    model, tokenizer = init_BaiChuan(MODEL_PATH, LoRA_PATH, logger)
elif "ChatGLM" in BASE_MODEL:
    model, tokenizer = init_GLM(MODEL_PATH, LoRA_PATH, logger)


# 创建一个POST请求的API端点
@app.post("/model/")
async def create_item(item: Item):
    # 实现音频文件处理
    prompt = item.prompt
    temperature = item.temperature
    max_length = item.max_length
    if "BaiChuan" in BASE_MODEL:
        try:
            logger.debug("用户提问：{}，温度设置为{}，最大长度设置为：{}".format(prompt, temperature, max_length))
            response = get_baichuan_completion(model, tokenizer, prompt, temperature, max_length)
            logger.debug("BaiChuan回复：{}".format(response))
        except Exception as e:
            logger.error("调用 API 出现错误，错误信息为：{}".format(e))
            raise e
    elif "ChatGLM" in BASE_MODEL:
        try:
            logger.debug("用户提问：{}，温度设置为{}，最大长度设置为：{}".format(prompt, temperature, max_length))
            response = get_glm_completion(model, tokenizer, prompt, temperature, max_length)
            logger.debug("GLM回复：{}".format(response))
        except Exception as e:
            logger.error("调用 API 出现错误，错误信息为：{}".format(e))
            raise e
    return response


