# Chat-嬛嬛

**Chat-甄嬛**是利用《甄嬛传》剧本中所有关于甄嬛的台词和语句，基于**ChatGLM2**进行**LoRA微调**得到的模仿甄嬛语气的聊天语言模型。

> 甄嬛，小说《后宫·甄嬛传》和电视剧《甄嬛传》中的女一号，核心女主角。原名甄玉嬛，嫌玉字俗气而改名甄嬛，为汉人甄远道之女，后被雍正赐姓钮祜禄氏，抬旗为满洲上三旗，获名“钮祜禄·甄嬛”。同沈眉庄、安陵容参加选秀，因容貌酷似纯元皇后而被选中。入宫后面对华妃的步步紧逼，沈眉庄被冤、安陵容变心，从偏安一隅的青涩少女变成了能引起血雨腥风的宫斗老手。雍正发现年氏一族的野心后令其父甄远道剪除，甄嬛也于后宫中用她的连环巧计帮皇帝解决政敌，故而深得雍正爱待。几经周折，终于斗垮了嚣张跋扈的华妃。甄嬛封妃时遭皇后宜修暗算，被皇上嫌弃，生下女儿胧月后心灰意冷，自请出宫为尼。然得果郡王爱慕，二人相爱，得知果郡王死讯后立刻设计与雍正再遇，风光回宫。此后甄父冤案平反、甄氏复起，她也生下双生子，在滴血验亲等各种阴谋中躲过宜修的暗害，最后以牺牲自己亲生胎儿的方式扳倒了幕后黑手的皇后。但雍正又逼甄嬛毒杀允礼，以测试甄嬛真心，并让已经生产过孩子的甄嬛去准格尔和亲。甄嬛遂视皇帝为最该毁灭的对象，大结局道尽“人类的一切争斗，皆因统治者的不公不义而起”，并毒杀雍正。四阿哥弘历登基为乾隆，甄嬛被尊为圣母皇太后，权倾朝野，在如懿传中安度晚年。

项目最主要目的是学习***transformers***和*大模型微调技术*，目前LoRA微调技术参考[ChatGLM-Efficient-Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning)项目，欢迎给原作者项目star，所使用的[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)模型也欢迎大家前去star。

## 使用方法

### 环境安装

首先下载本仓库，再用pip安装环境依赖：

```shell
git clone https://github.com/KMnO4-zx/huanhuan-chat.git
cd ./huanhuan-chat
pip install -r requirements.txt
```

### 代码调用

```python
>>> from peft import PeftModel
>>> from transformers import AutoTokenizer, AutoModel
>>> model_path = "THUDM/chatglm2-6b"
>>> model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
>>> tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
>>> #  给你的模型加上嬛嬛LoRA! 
>>> model = PeftModel.from_pretrained(model, "lora/sft").half()
>>> model.eval()
>>> response, history = model.chat(tokenizer, "你好", history=history)
>>> print(response)
```

```
皇上好，我是甄嬛，家父是大理寺少卿甄远道。
```

### 网页demo

首先需要到[Hugging Face Hub](https://huggingface.co/THUDM/chatglm2-6b)下载ChatGLM2-6B的模型文件，然后替换`scrupt/web_demo.py`中的`model_path`为你下载的模型地址，然后运行下面的命令：

```
python ./script/web_demo.py
```

## ToDo

初版：

- [x] 基于《甄嬛传》剧本、ChatGLM2、Lora 微调得到初代的chat-甄嬛聊天模型。

数据优化：

- [ ] 结合 ChatGPT API，优化训练问答对
- [ ] 基于《后宫甄嬛传》原著小说，构建训练问答对（基于向量数据库及图数据库模式）

模型优化：

- [ ] 使用多种微调方法对ChatGLM2训练微调，找到最适合聊天机器人的微调方法。
- [ ] 尝试多种开源大模型（Baichuan、ChatGLM等），找到效果最好的开源大模型

应用优化：

- [ ] 实现语音与甄嬛对话，生成数字人甄嬛
- [ ] 实现支持并发、高可用性部署
- [ ] 提升推理速度
- [ ] 优化开发前后端
- [ ] 使用Langchain与huanhuan-chat结合。

最终目标：

- [ ] 实现普及版流程，支持对任意一本小说、电视剧生成数据集，训练***个性化AI —— character AI！***）

## 案例展示

![侍寝](image/侍寝.png)

![晚上有些心累](image/晚上有些心累.png)

![午饭吃什么](image/午饭吃什么.png)

## 赞助

接受私人定制

（二维码区域）
