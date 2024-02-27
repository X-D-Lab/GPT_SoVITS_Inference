### 项目描述
将GPT_SoVITS项目中的推理部分提取并暴露出来。\
从modelscope： XD-LAB获取语音模型，进行tts文本转语音推理。
### 文件树
```
-gpt_sovits_tts
    -GPT_SoVITS 推理使用的代码
    -temp 未指定推理结果wav存放路径时，临时存放推理结果
    -get_tts_wav.py 提供推理类GPT_SoVITS_TTS_inference(),具体使用方法见example.py
    -example.py 调用示例
```

使用时请将文件夹gpt_sovits_tts存放在项目根目录下
推理所占显存约2g
### 安装依赖

python版本建议3.9,3.10

#### pip
windows
```
pip install -r requirements_win.txt
```
#### conda

windows：
```
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
不需要 ffmpeg.exe和 ffprobe.exe也能进行推理
注，推理时还需要nltk的一些数据，仅在推理时自动下载（需梯子），如果装不了，把temp文件夹下的nltk_data文件夹
复制粘贴到conda环境中和Lib同级目录下

Linux：
直接运行install.sh

### 致谢
https://github.com/RVC-Boss/GPT-SoVITS
