
## 👏 项目描述

原始[GPT_SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)的效果体验和推理服务较大依赖于基于Gradio的webui界面，为了更方便地推理体验GPT_SoVITS效果，本项目将其推理部分提取病暴露出来，支持一键式的推理部署。

## 🔥 模型列表

| 模型名称 | 模型下载 | 角色特点 | 语言 |
| :----: | :----: | :----: | :----: |
| TTS-GPT_SoVITS-sunshine_girl | [🤗]() / [🤖](https://modelscope.cn/models/X-D-Lab/TTS-GPT_SoVITS-sunshine_girl/summary) | 阳光少女 | zh |
| TTS-GPT_SoVITS-heartful_sister | [🤗]() / [🤖](https://modelscope.cn/models/X-D-Lab/TTS-GPT_SoVITS-heartful_sister/summary) | 知性姐姐 | zh |

- 预训练模型

| 模型名称 | 模型下载 |
| :----: | :----: |
| GPT-SoVITS | [🤗]() / [🤖](https://modelscope.cn/models/X-D-Lab/TTS-GPT_SoVITS-pretrained_models/summary) |


## ⚒️ 安装依赖

推荐 Python>=3.9,<=3.10

```
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

git clone https://github.com/X-D-Lab/GPT_SoVITS_Inference.git
cd GPT_SoVITS_Inference
pip install -r requirements.txt
```
如果您是windows使用者，请下载并将 [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) 和 [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe) 放置在本项目的根目录下。


## 😇 如何使用

详细内容可以参见[example.py](./example.py)

```Python

import os
import sys

project_root = os.path.abspath('.')
sys.path.append(project_root)


from get_tts_wav import GPT_SoVITS_TTS_inference

text = """我是MindChat漫谈心理大模型"""

inference = GPT_SoVITS_TTS_inference(prompt_language='zh', base_model_id='X-D-Lab/TTS-GPT_SoVITS-pretrained_models', audio_model_id='X-D-Lab/TTS-GPT_SoVITS-sunshine_girl')

inference.get_tts_wav(text=text, wav_save_path="./temp/output1.wav")

```
## 👏 Contributors
本项目仍然属于非常早期的阶段，欢迎各位开发者加入！

<a href="https://github.com/thomas-yanxin/Sunsimiao/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=X-D-Lab/GPT_SoVITS_Inference" />
</a>  

### 🙇‍ 致谢

本项目基于[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)进行，感谢他们的开源贡献。
