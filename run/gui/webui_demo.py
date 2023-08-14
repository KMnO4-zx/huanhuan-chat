#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   webui_demo.py
@Time    :   2023/08/14 17:15:05
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   Chat-huanhuan GUI 部署
'''

import torch
import mdtex2html
import gradio as gr

from peft import PeftModel
from transformers import AutoTokenizer, AutoModel, GenerationConfig, AutoModelForCausalLM, HfArgumentParser
from dataclasses import dataclass, field
import sys
# 导入 log 模块目录
sys.path.append("../../")
from log.logutli import Logger

@dataclass
class RunArguments:
    # 运行参数
    model_path: str = field(default = "../../dataset/model")
    base_model: str = field(default = "ChatGLM2")
    lora_path: str = field(default = "../../dataset/output")
    log_name: str = field(default="log")

# Parse 命令行参数
run_args = HfArgumentParser(RunArguments).parse_args_into_dataclasses()[0]

# 日志设置
log_id     = 'run_gui'  
log_dir    = f'../../log/'
log_name   = '{}.log'.format(run_args.log_name)
log_level  = 'debug'

# 初始化日志
logger = Logger(log_id, log_dir, log_name, log_level).logger
logger.info('部署图形化界面')

logger.debug("命令行参数")
logger.debug("run_args:")
logger.debug(run_args.__repr__())

if "ChatGLM" in run_args.base_model:
    model = AutoModel.from_pretrained(
        run_args.model_path, trust_remote_code=True).half().cuda()
    logger.info("从{}加载模型成功".format(run_args.model_path))
elif "BaiChuan" in run_args.base_model:
    model = AutoModelForCausalLM.from_pretrained(
        run_args.model_path, trust_remote_code=True).half().cuda()
    logger.info("从{}加载模型成功".format(run_args.model_path))
else:
    logger.error("错误参数：底座模型必须是 ChatGLM 或者 BaiChuan")
    raise ValueError("错误参数：底座模型必须是 ChatGLM 或者 BaiChuan")

tokenizer = AutoTokenizer.from_pretrained(run_args.model_path, trust_remote_code=True)
logger.info("从{}加载 tokenizer 成功".format(run_args.model_path))

model = PeftModel.from_pretrained(model, run_args.lora_path).half()
logger.info("加载 LoRa 参数成功")


"""重载聊天机器人的数据预处理"""
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

gr.Chatbot.postprocess = postprocess

'''文本处理'''
def parse_text(text):  # copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

# GLM 使用
def predict(input, chatbot, max_length, top_p, temperature, history):
    logger.debug("用户输入：{}".format(input))
    chatbot.append((parse_text(input), ""))
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))
        logger.debug("模型回答：{}".format(response))
        yield chatbot, history

# BaiChuan 使用
def generate(input_text, chatbot, max_length, top_p, temperature):
    logger.debug("用户输入：{}".format(input_text))
    chatbot.append((parse_text(input_text), ""))
    prompt = "Human: " + input_text + "\n\nAssistant: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=2.0,
        max_new_tokens=max_length,  # max_length=max_new_tokens+input_sequence

    )
    generate_ids = model.generate(**inputs, generation_config=generation_config)
    output = tokenizer.decode(generate_ids[0][len(inputs.input_ids[0]):])
    logger.debug("模型回答：{}".format(output))
    chatbot[-1] = (parse_text(input_text), parse_text(output))
    return chatbot, None, None


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []

# 实现图形化界面
if model != None:
        
    with gr.Blocks() as demo:

        gr.HTML("""
        <h1 align="center">
                Chat-嬛嬛 v2.0
        </h1>
        """)

        chatbot = gr.Chatbot()

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(
                        show_label=False, placeholder="Input...", lines=10).style(container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")

            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(
                    0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.01,
                                label="Top P", interactive=True)
                temperature = gr.Slider(
                    0, 1.5, value=0.95, step=0.01, label="Temperature", interactive=True)

        history = gr.State([])

        # BaiChuan
        if "BaiChuan" in run_args.base_model:
            submitBtn.click(generate, [user_input, chatbot, max_length, top_p, temperature], [
                            chatbot, history], show_progress=True)
        # GLM
        if "ChatGLM" in run_args.base_model:
            submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [
                        chatbot, history], show_progress=True)
            
        submitBtn.click(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

    demo.queue().launch(server_name="0.0.0.0", share=False,
                        inbrowser=False, server_port=35898)