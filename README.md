# Chat-嬛嬛

**Chat-甄嬛**是利用《甄嬛传》剧本中所有关于甄嬛的台词和语句，基于**ChatGLM2**进行**LoRA微调**得到的模仿甄嬛语气的聊天语言模型。

> 甄嬛，小说《后宫·甄嬛传》和电视剧《甄嬛传》中的女一号，核心女主角。原名甄玉嬛，嫌玉字俗气而改名甄嬛，为汉人甄远道之女，后被雍正赐姓钮祜禄氏，抬旗为满洲上三旗，获名“钮祜禄·甄嬛”。同沈眉庄、安陵容参加选秀，因容貌酷似纯元皇后而被选中。入宫后面对华妃的步步紧逼，沈眉庄被冤、安陵容变心，从偏安一隅的青涩少女变成了能引起血雨腥风的宫斗老手。雍正发现年氏一族的野心后令其父甄远道剪除，甄嬛也于后宫中用她的连环巧计帮皇帝解决政敌，故而深得雍正爱待。几经周折，终于斗垮了嚣张跋扈的华妃。甄嬛封妃时遭皇后宜修暗算，被皇上嫌弃，生下女儿胧月后心灰意冷，自请出宫为尼。然得果郡王爱慕，二人相爱，得知果郡王死讯后立刻设计与雍正再遇，风光回宫。此后甄父冤案平反、甄氏复起，她也生下双生子，在滴血验亲等各种阴谋中躲过宜修的暗害，最后以牺牲自己亲生胎儿的方式扳倒了幕后黑手的皇后。但雍正又逼甄嬛毒杀允礼，以测试甄嬛真心，并让已经生产过孩子的甄嬛去准格尔和亲。甄嬛遂视皇帝为最该毁灭的对象，大结局道尽“人类的一切争斗，皆因统治者的不公不义而起”，并毒杀雍正。四阿哥弘历登基为乾隆，甄嬛被尊为圣母皇太后，权倾朝野，在如懿传中安度晚年。

本项目预计以《甄嬛传》为切入点，打造一套基于小说、剧本的**个性化 AI** 微调大模型完整流程，目标是让每一个人都可以基于心仪的小说、剧本微调一个属于自己的、契合小说人设、能够流畅对话的个性化大模型。

目前，本项目已实现分别基于 ChatGLM2、BaiChuan 等大模型，使用 LoRA 微调的多版本 Chat-甄嬛，具备甄嬛人设，欢迎大家体验交流~目前LoRA微调技术参考[ChatGLM-Efficient-Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning)项目，欢迎给原作者项目star，所使用的[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)模型、[BaiChuan](https://github.com/baichuan-inc/Baichuan-7B)模型也欢迎大家前去star。

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

### 网页 demo

首先需要到[Hugging Face Hub](https://huggingface.co/THUDM/chatglm2-6b)下载ChatGLM2-6B的模型文件，然后替换`script/web_demo.py`中的`model_path`为你下载的模型地址，然后运行下面的命令：

```
python ./script/web_demo.py
```
网页 Demo 默认使用以 ChatGLM2-6B 为底座的 Chat-甄嬛-GLM，如果你想使用以 BaiChuan7B 为底座的 Chat-甄嬛-BaiChuan，请同样下载 BaiChuan7B 的模型文件并替换模型路径，并将 `script/web_demo.py` 中的 `model_name` 替换成 'BaiChuan'，然后运行上述命令。

### LoRA 微调
如果你想本地复现 Chat-甄嬛，直接运行微调脚本即可：

```
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --model_name_or_path THUDM/chatglm2-6b \
    --stage sft \
    --use_v2 \
    --do_train \
    --dataset zhenhuan \
    --finetuning_type lora \
    --lora_rank 8 \
    --output_dir ./output/sft \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 4.0 \
    --fp16
```
如果你想使用自定义数据集，请根据 [甄嬛对话集](/data/zhenhuan.json) 的数据集格式构建类似对话集并存放在对应目录下，修改 dataset 参数即可。**后续我们将提供从指定小说或剧本一站式构建对话集的脚本，敬请关注**。

如果你想使用本地已下载的 ChatGLM2-6B 模型，修改 model_name_or_path 即可。

如果你想尝试基于 BaiChuan-7B 微调，请运行以下命令：

```
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
```

限于目前代码还比较粗糙，修改模型路径、数据集路径等参数，可在 train_baichuan.py 中修改全局变量。

## News

[2023.07.14]：完成 BaiChuan 模型训练及 web 调用，完成初步语音支持及数据集处理。

[2023.07.12]：完成RM、RLHF训练（存在问题），新的小伙伴加入项目。

[2023.07.11]：优化数据集，解决prompt句末必须携带标点符号的问题。

[2023.07.09]：完成初次LoRA训练。

## Edition

V1.0：

- [x] 基于《甄嬛传》剧本、ChatGLM2、Lora 微调得到初代的chat-甄嬛聊天模型。

V1.1:

- [ ] 基于优化数据集、优化训练方法、支持甄嬛语音的新版Chat-甄嬛聊天模型。

## To do

- [x] 实现V1.0Chat-甄嬛的训练及部署

- [ ] 数据集生成流程实现
    - [ ] 利用gpt从甄嬛传小说中提取特色对话集。
    - [ ] 优化甄嬛传剧本提取对话集。
    - [ ] 基于hugging face上日常对话数据集+GPT prompt+Langchain 生成个性化日常对话数据集
    - [ ] 探究生成多轮对话数据集

- [ ] 探索更多元的 Chat-甄嬛
    - [ ] 使用多种微调方法对ChatGLM2训练微调，找到最适合聊天机器人的微调方法。
    - [ ] 尝试多种开源大模型（Baichuan13B、ChatGLM等），找到效果最好的开源大模型
    - [ ] 寻找微调的最优参数

- [ ] 打造更智能的 Chat-甄嬛
    - [ ] 实现语音与甄嬛对话，生成数字人甄嬛
    - [ ] 实现支持并发、高可用性部署
    - [ ] 提升推理速度
    - [ ] 优化开发前后端
    - [ ] 使用Langchain与huanhuan-chat结合。

- [ ] 打造**个性化微调大模型通用流程**！ 

## 案例展示

![侍寝](image/侍寝.png)

![晚上有些心累](image/晚上有些心累.png)

![午饭吃什么](image/午饭吃什么.png)

## 人员贡献

[不要葱姜蒜](https://github.com/KMnO4-zx)：整理数据集，完成SFT训练。

[Logan Zou](https://github.com/nowadays0421)：完成 BaiChuan 训练及调用。

[coderdeepstudy](https://github.com/coderdeepstudy)：Window环境下的Lora微调，服务器支持。

[Bald0Wang](https://github.com/Bald0Wang)：完成甄嬛语音支持。
## 赞助

如果您愿意请我们喝一杯咖啡，帮助我们打造更美丽的甄嬛，那就再好不过了~

![赞助](image/赞助.jpg)

另外，如果您有意向，我们也接受私人定制，欢迎联系本项目负责人[不要葱姜蒜](https://github.com/KMnO4-zx)

