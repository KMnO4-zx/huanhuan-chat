# Extract Dialogue

>***本仓库只为`huanhuan-chat`泛化版的一部分内容（文本对话抽取），欢迎大家给`huanhuan-chat`仓库star！本仓库的最大贡献就是为泛化的Character AI提供了从小说中建立数据集的功能。***
>
>`huanhuan-chat: https://github.com/KMnO4-zx/huanhuan-chat.git`

## Show

`repo`：https://github.com/KMnO4-zx/extract-dialogue.git

本项目利用`chatgpt`从小说中提取对话集，提取的样本中包括`role`，`dialogue`，比如以下的形式：

```json
{
    "role": "艾伦",
    "dialogue": "不，不要提，这真是太倒霉了！我从楼梯上摔了下去，出现了较为严重的骨裂，只能打石膏做固定。"
}
{
    "role": "克莱恩",
    "dialogue": "真是不够走运啊。"
}
```

## QuickStart

- 克隆仓库并切换目录：`git clone https://github.com/KMnO4-zx/extract-dialogue.git `，`cd extract-dialogue`

- 安装依赖：`pip install -r requirements.txt`
- 在当前目录创建`.env`文件，并填入`OPENAI_API_KEY`。（***需要一个不限制请求次数的key，openai对于免费帐户的api限制为：每分钟请求至多请求三次，每天至多请求200次。对于从小说中提取对话来说，一天200次杯水车薪。***）
- 把你要提取的小说或文本，放到当前目录，在`main.py`中修改`path`。
- ***强烈建议您结合要提取的小说修改`main.py`中的`schema`示例。在下面的部分中有详细介绍`schema`。***
- 修改main.py中的roles列表，再列表中填写要提取的角色的名称。比如在《西游记》中要提取悟空的对话，那就需要把所有有关孙悟空的名称写上。

```python
roles = ['孙悟空', '悟空', '石猴', '美猴王', '孙大圣', '齐天大圣']
```

- 运行`main.py`，`python main.py`

> ***注意：gpt3.5收费标准2000token 0.002美元，提取一本小说花费大概在5美刀。***

## Introduction

`Kor`是个支持`LangChain`的文本抽取库，可以把文本抽取成`json`格式。

我们来简单使用一下`Kor`，首先用`langchain`的`LLM`模块重新封装一下，`langchian`中的`ChatOpenAI`类。我曾尝试在两台电脑上使用`Kor`都因`ChatOpenAI`类报错而以失败告终，所以大家直接不走弯路！

当然要使用`chatgpt`你需要用一个`api key`，然后在当前目录创建一个`.env`文件，并在其中填写：

```python
OPENAI_API_KEY=your key
```

`your key`当然要替换为你自己的`api key`喽~

```python
import openai
import os
from langchain.llms.base import LLM
from typing import Dict, List, Optional, Tuple, Union
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]

def get_completion(prompt, model="gpt-3.5-turbo"):
    '''
    prompt: 对应的提示词
    model: 调用的模型，默认为 gpt-3.5-turbo(ChatGPT)，有内测资格的用户可以选择 gpt-4
    '''
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # 模型输出的温度系数，控制输出的随机程度
    )
    # 调用 OpenAI 的 ChatCompletion 接口
    return response.choices[0].message["content"]

class OpenAI_LLM(LLM):
    model_type: str = "openai"


    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "openai"
    
    def _call(self, prompt: str, history: List = [], stop: Optional[List[str]] = None) -> str:
        res = get_completion(prompt)
        return res
    
    @property
    def _identifying_params(self):
        """Get the identifying parameters.
        """
        _param_dict = {
            "model": self.model_type
        }
        return _param_dict
```

`Kor`的核心就要写一段对你所提取信息的描述，以及几个示例，在下面的例子中（我挑取了小说《诡秘之主》中的一个片段），我所要提取的信息分别是`role`，`dialogue`，也就是角色和其对应的台词。并且我给出了几个示例。

- 信息描述：

```
[
        Text(
            id="role",
            description="The character who is speaking",
        ),
        Text(
            id="dialogue",
            description="The dialogue spoken by the characters in the sentence",
        )
]
```

- 输出示例：

```
(
            '''
            他下意识放轻了脚步，不制造出明显的噪音。
            刚登上二楼，他看见盥洗室的门突然打开，穿着旧布长裙的梅丽莎一副睡眼惺忪的模样出来。
            “你回来了……”梅丽莎还有些迷糊地揉了揉眼睛。
            克莱恩掩住嘴巴，打了个哈欠道：
            “是的，我需要一个美好的梦境，午餐之前都不要叫醒我。”
            梅丽莎“嗯”了一声，忽然想起什么似地说道：
            “我和班森上午要去圣赛琳娜教堂做祈祷，参与弥撒，午餐可能会迟一点。”
            ''',
            [
                {"role": "梅丽莎", "dialogue": "你回来了……"},
                {"role": "克莱恩", "dialogue": "是的，我需要一个美好的梦境，午餐之前都不要叫醒我。"},
                {"role": "梅丽莎", "dialogue":"我和班森上午要去圣赛琳娜教堂做祈祷，参与弥撒，午餐可能会迟一点。"}
            ],
)
```

- 完整代码

```py
schema = Object(
    id="script",
    description="Adapted from the novel into script",
    attributes=[
        Text(
            id="role",
            description="The character who is speaking",
        ),
        Text(
            id="dialogue",
            description="The dialogue spoken by the characters in the sentence",
        )
    ],
    examples=[
        (
            '''
            他下意识放轻了脚步，不制造出明显的噪音。
            刚登上二楼，他看见盥洗室的门突然打开，穿着旧布长裙的梅丽莎一副睡眼惺忪的模样出来。
            “你回来了……”梅丽莎还有些迷糊地揉了揉眼睛。
            克莱恩掩住嘴巴，打了个哈欠道：
            “是的，我需要一个美好的梦境，午餐之前都不要叫醒我。”
            梅丽莎“嗯”了一声，忽然想起什么似地说道：
            “我和班森上午要去圣赛琳娜教堂做祈祷，参与弥撒，午餐可能会迟一点。”
            ''',
            [
                {"role": "梅丽莎", "dialogue": "你回来了……"},
                {"role": "克莱恩", "dialogue": "是的，我需要一个美好的梦境，午餐之前都不要叫醒我。"},
                {"role": "梅丽莎", "dialogue":"我和班森上午要去圣赛琳娜教堂做祈祷，参与弥撒，午餐可能会迟一点。"}
            ],
        ),
        (
            '''
            “太感谢您了！‘愚者’先生您真是太慷慨了！”奥黛丽欣喜地回应道。
            她为自己刚才想用金钱购买消息的庸俗忏悔了三秒。
            克莱恩停止手指的敲动，语气平淡地描述道：
            “第一个常识，非凡特性不灭定律，非凡特性不会毁灭，不会减少，只是从一个事物转移到另一个事物。”
            我不知不觉竟然用上了队长的口吻……克莱恩的嘴角下意识就翘了起来。
            ''',
            [
                {"role": "奥黛丽", "dialogue": "太感谢您了！‘愚者’先生您真是太慷慨了！"},
                {"role": "克莱恩", "dialogue": "第一个常识，非凡特性不灭定律，非凡特性不会毁灭，不会减少，只是从一个事物转移到另一个事物。"},
            ],
        )
    ],
    many=True,
)
```

我们可以看一眼`Kor`的核心`Prompt`：

````python
llm = OpenAI_LLM()
chain = create_extraction_chain(llm, schema)
print(chain.prompt.format_prompt(text="[user input]").to_string())


Your goal is to extract structured information from the user's input that matches the form described below. When extracting information please make sure it matches the type information exactly. Do not add any attributes that do not appear in the schema shown below.

```TypeScript

script: Array<{ // Adapted from the novel into script
 role: string // The character who is speaking
 dialogue: string // The dialogue spoken by the characters in the sentence
}>
```


Please output the extracted information in CSV format in Excel dialect. Please use a | as the delimiter. 
 Do NOT add any clarifying information. Output MUST follow the schema above. Do NOT add any additional columns that do not appear in the schema.



Input: 
            他下意识放轻了脚步，不制造出明显的噪音。
            刚登上二楼，他看见盥洗室的门突然打开，穿着旧布长裙的梅丽莎一副睡眼惺忪的模样出来。
            “你回来了……”梅丽莎还有些迷糊地揉了揉眼睛。
            克莱恩掩住嘴巴，打了个哈欠道：
            “是的，我需要一个美好的梦境，午餐之前都不要叫醒我。”
            梅丽莎“嗯”了一声，忽然想起什么似地说道：
            “我和班森上午要去圣赛琳娜教堂做祈祷，参与弥撒，午餐可能会迟一点。”
            
...
克莱恩|第一个常识，非凡特性不灭定律，非凡特性不会毁灭，不会减少，只是从一个事物转移到另一个事物。

Input: [user input]
Output:
````

可以看到`Kor`的抽取实际上是让语言文字根据`input`输出了`csv`格式，OK，大家如果感兴趣可以直接去看当前目录下的`jupuyer`文件。

## 参考

【1】https://eyurtsev.github.io/kor/index.html#

【2】https://zhuanlan.zhihu.com/p/646948797