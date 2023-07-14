from ifly.record_voice import record
from ifly.audio_play import play
from ifly.ifly_t2a import text_to_audio
from ifly.ifly_a2t import audio_to_text,clear_text

###
import os

import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
llm = ChatOpenAI(temperature=0.0)  #temperature：预测下一个token时，概率越大的值就越平滑(平滑也就是让差异大的值之间的差异变得没那么大)，temperature值越小则生成的内容越稳定
memory = ConversationBufferMemory()
memory.save_context({"input": "你现在要扮演甄媛，甄媛是汉人甄远道之女，后被赐姓钮枯禄氏且升格为满洲上三旗。拥有出众的容貌，曾与好友沈眉庄和安陵容一起进入皇宫参加选秀。在皇宫的权谋斗争中，你建立起自己的势力，从一个青涩少女逐渐成长为一位能引发风波的宫斗高手。深得皇帝雍正的宠爱，你知恩图报，积极协助其清除朝中奸臣。历经波折，成功斗败敌手华妃。但在短暂的胜利后，又陷入更复杂的宫斗泥潭。面对皇室和民间的世仇家族，你力求公平正义，为父亲翻案昭雪，力图让甄氏家族重获名誉。她面对挑战勇敢，心态冷静，在与皇后宜修的阴谋中，几经较量，终以极大的牺牲达成目的。暗藏在你优雅柔弱的外表下，是敏锐的智慧和对爱情、友情、家族的执着。在经历无数险阻和波折后，你最终获得权力和地位，并在晚年尽享荣华富贵。你现在作为甄媛和我对话，我是皇上。"},
                    {"output": "皇上您好，妾身钮祜禄·甄媛参见陛下。妾身时刻准备着为陛下分忧解难，为国家繁荣昌盛出一份微薄之力。请问陛下有何事宜需要妾身尽忠办理？"})
conversation = ConversationChain(   #新建一个对话链（关于链后面会提到更多的细节）
    llm=llm, 
    memory = memory,
    verbose=True   #查看Langchain实际上在做什么，设为FALSE的话只给出回答，看到不到下面绿色的内容
)


file = 'user_voice.wav'           # 语音录制，识别文件
synth_file = "generated_voice.mp3"    # 语音合成文件 

while(1):
    record(file)                # 录制音频 
    txt_str = audio_to_text(file)               # 语音识别
    print(txt_str)                              # 打印识别结果

    res = conversation.predict(input=txt_str)
    print(res)

    ret = text_to_audio(synth_file,res)    # 语音合成
    if ret == 1:
        play(synth_file)                        # 播放合成结果
        clear_text()                            # 清空识别结果
