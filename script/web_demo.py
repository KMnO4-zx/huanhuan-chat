# coding=utf-8
# Implements user interface in browser for ChatGLM fine-tuned with PEFT.
# This code is largely borrowed from https://github.com/THUDM/ChatGLM-6B/blob/main/web_demo.py
# Usage: python web_demo.py --checkpoint_dir path_to_checkpoint [--quantization_bit 4]


import torch
import mdtex2html
import gradio as gr

from peft import PeftModel
from transformers import AutoTokenizer, AutoModel, GenerationConfig, AutoModelForCausalLM

model_path = "THUDM/chatglm2-6b"
model_name = "ChatGLM"

if model_name == "ChatGLM":
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True).half().cuda()
elif model_name == "BaiChuan":
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True).half().cuda()
else:
    print("调用不支持的模型")
    model = None

print("基座模型加载成功")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

#  给你的模型加上嬛嬛LoRA! output: lora存放路径
if model_name == "ChatGLM":
    lora_path = "../output/sft"
elif model_name == "BaiChuan":
    lora_path = "../output/baichuan-sft"

model = PeftModel.from_pretrained(model, lora_path).half()

print("成功加载LoRa")

"""Override Chatbot.postprocess"""


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
    chatbot.append((parse_text(input), ""))
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))

        yield chatbot, history

# BaiChuan 使用
def generate(input_text, chatbot, max_length, top_p, temperature, history):
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
    chatbot[-1] = (parse_text(input_text), parse_text(output))
    return chatbot, None, None


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []

if model != None:
        
    with gr.Blocks() as demo:

        gr.HTML("""
        <h1 align="center">
                Chat-嬛嬛 ChatGLM2
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
        if model_name == "BaiChuan":
            submitBtn.click(generate, [user_input, chatbot, max_length, top_p, temperature, history], [
                            chatbot, history], show_progress=True)
        # GLM
        if model_name == "ChatGLM":
            submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [
                        chatbot, history], show_progress=True)
            
        submitBtn.click(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

    demo.queue().launch(server_name="0.0.0.0", share=False,
                        inbrowser=False, server_port=35898)
