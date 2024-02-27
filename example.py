# 可以在任意地方跨目录调用get_tts_wav()
"""
# ===关于推理文本的语种 参考===
# 在config和调用get_tts_wav时，对于prompt_language和text_language参数
    "all_zh"    #全部按中文识别
    "en"        #全部按英文识别#######不变
    "all_ja"    #全部按日文识别
    "zh"        #按中英混合识别####不变
    "ja"        #按日英混合识别####不变
    "auto"      #多语种混合，启动切分识别语种
}
"""
"""
def get_tts_wav(
    text: str,      # 要转换为语音的文本。get_tts_wav()内部会对文本按标点自动切割。
    text_language: str = "zh", # 推理出的语音语言，暂不需要修改
    wav_savepath: str = "temp/output.wav" # 推理结果存放的路径与文件名称。会得到一个完整的wav
    ==其他次要参数==
    how_to_cut: str = "凑四句一切", # 切割推理文本的方法，一共有5种。
            # 推荐"凑四句一切"和"按标点符号切"。"按标点符号切"语速最慢,推理最准确
            # "凑四句一切","凑50字一切","按中文句号。切","按英文句号.切","按标点符号切"
    top_k: int = 20,
    top_p: float = 0.6,
    temperature: float = 0.6,
            # 关于上面三个参数 https://github.com/RVC-Boss/GPT-SoVITS/pull/457
    ref_free: bool = False  # 不输入参考音频内对应文本，进行推理。默认关闭
) -> None
"""
import os
import sys

project_root = os.path.abspath('.')
sys.path.append(project_root)

import time

# from gpt_sovits_tts.get_tts_wav import GPT_SoVITS_TTS_inference
from get_tts_wav import GPT_SoVITS_TTS_inference

text2 = """我是MindChat漫谈心理大模型"""
text3 = """这是一个长文本示例。它包含多个句子，每个句子都以标点符号结束！
这样可以测试split函数的效果。希望它能正确分割文本？"""
text5 = """生活就像海洋，只有意志坚强的人，才能到达彼岸。"Life is like the ocean. Only those with strong wills can reach the other shore."
"""
text6 = """当然可以！首先，请告诉我你目前的压力源是什么？"""
start_time_1 = time.time()

"""
# 目前[20240227]modelscope上可用的语音模型audio_model_id
X-D-Lab/TTS-GPT_SoVITS-sunshine_girl
X-D-Lab/TTS-GPT_SoVITS-heartful_sister
"""

inference = GPT_SoVITS_TTS_inference(prompt_language='zh', base_model_id='X-D-Lab/TTS-GPT_SoVITS-pretrained_models', audio_model_id='X-D-Lab/TTS-GPT_SoVITS-sunshine_girl')

print("初始化时间：",time.time()-start_time_1)
start_time_2 = time.time()
inference.get_tts_wav(text=text2, wav_save_path="./temp/output1.wav")
end_time_2 = time.time()
print("推理时间1：",end_time_2 -start_time_2)
start_time_3 = time.time()
inference.get_tts_wav(text=text3, wav_save_path="./temp/output2.wav")
end_time_3 = time.time()
print("推理时间2：",end_time_3-start_time_3)

start_time_4 = time.time()
inference.get_tts_wav(text=text5, wav_save_path="./temp/output3.wav")
end_time_4 = time.time()
print("推理时间3：",end_time_4-start_time_4)
start_time_5 = time.time()
inference.get_tts_wav(text=text6, wav_save_path="./temp/output4.wav")
end_time_5 = time.time()
print("推理时间4：",end_time_5 -start_time_5)