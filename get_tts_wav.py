import os
import re
# import signal
import sys
from time import time as ttime

import LangSegment
import librosa
import numpy as np
import soundfile as sf
import torch
from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModelForMaskedLM, AutoTokenizer

from __init__ import root_dir
from GPT_SoVITS.AR.models.t2s_lightning_module import \
    Text2SemanticLightningModule
from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.module.mel_processing import spectrogram_torch
# from io import BytesIO
from GPT_SoVITS.module.models import SynthesizerTrn
from GPT_SoVITS.my_utils import load_audio
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

# =====================

dict_language = {
    "中文": "all_zh",#全部按中文识别
    "英文": "en",#全部按英文识别#######不变
    "日文": "all_ja",#全部按日文识别
    "中英混合": "zh",#按中英混合识别####不变
    "日英混合": "ja",#按日英混合识别####不变
    "多语种混合": "auto",#多语种启动切分识别语种
    "ZH": "zh",
    "EN": "en",
    "JA": "ja",
    "zh": "zh",
    "en": "en",
    "ja": "ja",
    "all_zh": "all_zh", #手动添加，以防万一
    "all_ja": "all_ja", #手动添加，以防万一
    "auto": "auto" #手动添加，以防万一
}

splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}  # 不考虑省略号

# ====== 识别与切割混合语种的推理文本 ======

def splite_en_inf(sentence, language):
    pattern = re.compile(r'[a-zA-Z ]+')
    textlist = []
    langlist = []
    pos = 0
    for match in pattern.finditer(sentence):
        start, end = match.span()
        if start > pos:
            textlist.append(sentence[pos:start])
            langlist.append(language)
        textlist.append(sentence[start:end])
        langlist.append("en")
        pos = end
    if pos < len(sentence):
        textlist.append(sentence[pos:])
        langlist.append(language)
    # Merge punctuation into previous word
    for i in range(len(textlist)-1, 0, -1):
        if re.match(r'^[\W_]+$', textlist[i]):
            textlist[i-1] += textlist[i]
            del textlist[i]
            del langlist[i]
    # Merge consecutive words with the same language tag
    i = 0
    while i < len(langlist) - 1:
        if langlist[i] == langlist[i+1]:
            textlist[i] += textlist[i+1]
            del textlist[i+1]
            del langlist[i+1]
        else:
            i += 1

    return textlist, langlist

def clean_text_inf(text, language):
    formattext = ""
    language = language.replace("all_","")
    for tmp in LangSegment.getTexts(text):
        if language == "ja":
            if tmp["lang"] == language or tmp["lang"] == "zh":
                formattext += tmp["text"] + " "
            continue
        if tmp["lang"] == language:
            formattext += tmp["text"] + " "
    while "  " in formattext:
        formattext = formattext.replace("  ", " ")
    phones, word2ph, norm_text = clean_text(formattext, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text



def nonen_clean_text_inf(text, language):
    if(language!="auto"):
        textlist, langlist = splite_en_inf(text, language)
    else:
        textlist=[]
        langlist=[]
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    phones_list = []
    word2ph_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        phones_list.append(phones)
        if lang == "zh":
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)
    #【日志】 print(word2ph_list)
    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = ' '.join(norm_text_list)

    return phones, word2ph, norm_text

def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

# ====== 对输入文本进行切割 =========

def split(todo_text):
    """
    将大段文本按标点切割，并将每段文本(保留末尾标点)组成列表。
    """
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    """
    第一种文本分段法：基于重写的split分割后，凑4段语句推理一次。
    """
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)


def cut2(inp):
    """
    第二种文本分段法：基于重写split分割后，凑50个字推理一次。
    """
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return [inp]
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    if len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)

def cut3(inp):
    """
    第三种文本分段法：仅仅按中文句号分割。
    """
    inp = inp.strip("\n")
    return "\n".join(["%s。" % item for item in inp.strip("。").split("。")])

# 新增两种切法

def cut4(inp):
    """
    "按英文句号.切"
    """
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip(".").split(".")])


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    """
    "按标点符号切"
    """
    # if not re.search(r'[^\w\s]', inp[-1]):
    # inp += '。'
    inp = inp.strip("\n")
    punds = r'[,.;?!、，。？！;：…]'
    items = re.split(f'({punds})', inp)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    # 在句子不存在符号或句尾无符号的时候保证文本完整
    if len(items)%2 == 1:
        mergeitems.append(items[-1])
    opt = "\n".join(mergeitems)
    return opt


def load_model(cnhubert_base_path, bert_path, dict_s1, dict_s2, is_half, device):
    # 加载模型
    cnhubert.cnhubert_base_path = cnhubert_base_path
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
    if is_half:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)

    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"

    config = dict_s1["config"]
    ssl_model = cnhubert.get_model()
    if is_half:
        ssl_model = ssl_model.half().to(device)
    else:
        ssl_model = ssl_model.to(device)

    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    # 超长日志输出-missing_keys
    vq_model.load_state_dict(dict_s2["weight"], strict=False)

    t2s_model = Text2SemanticLightningModule(config, "ojbk", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()

    return tokenizer, bert_model, hps, config, ssl_model, vq_model, t2s_model

def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                            hps.data.win_length, center=False)
    return spec

def get_gpt_sovits_tts_models(model_id):

    if not os.path.exists(model_id):
        
        model_dir = snapshot_download(model_id)
    else:
        model_dir = model_id
    gpt_path = os.path.join(model_dir, [x for x in os.listdir(model_dir) if x.endswith('.ckpt')][0])
    sovits_path = os.path.join(model_dir, [x for x in os.listdir(model_dir) if x.endswith('.pth')][0])
    audio_refer_path = os.path.join(model_dir, [x for x in os.listdir(model_dir) if x.endswith('.wav')][0])
    audio_refer_text_path = os.path.join(model_dir, [x for x in os.listdir(model_dir) if x.endswith('.txt')][0])
    return gpt_path, sovits_path, audio_refer_path, audio_refer_text_path

def get_base_pretrained_models(model_id):
    if not os.path.exists(model_id):
        
        model_dir = snapshot_download(model_id)
    else:
        model_dir = model_id
    
    cnhubert_base_path = os.path.join(model_dir, [x for x in os.listdir(model_dir) if 'chinese-hubert-base' in x][0])
    bert_path = os.path.join(model_dir, [x for x in os.listdir(model_dir) if 'chinese-roberta-wwm-ext-large' in x][0])
    return cnhubert_base_path, bert_path



class GPT_SoVITS_TTS_inference(object):
    def __init__(self,  prompt_language, base_model_id, audio_model_id):
        self.prompt_language = prompt_language

        self.cnhubert_base_path, self.bert_path = get_base_pretrained_models(base_model_id)
        self.gpt_path, self.sovits_path, self.ref_wav_path, self.prompt_text_path = get_gpt_sovits_tts_models(audio_model_id)
        self.audio_model_id = audio_model_id

        self.is_half = True  # 半精度推理
        self.device = "cuda"
        self.n_semantic = 1024
        self.dict_s1 = torch.load(self.gpt_path, map_location="cpu")
        self.dict_s2 = torch.load(self.sovits_path, map_location="cpu")
        self.tokenizer, self.bert_model, self.hps, self.config, self.ssl_model, self.vq_model, self.t2s_model = load_model(
            self.cnhubert_base_path, self.bert_path, self.dict_s1, self.dict_s2, self.is_half, self.device)
        self.refer = get_spepc(self.hps, self.ref_wav_path)
        self.dtype=torch.float16 if self.is_half == True else torch.float32 #【补】

    # ======适配混合语种输出======
    # ===
    def get_cleaned_text_final(self,text,language):
        """
        根据语言类型选择适当的文本清洗函数，并返回处理后的音素序列、单词到音素的映射以及规范化文本。
        -> phones,word2ph,norm_text
            - clean_text_inf 针对单一语种{"en","all_zh","all_ja"}
                - clean_text 和 cleaned_text_to_sequence 来自内部text模块cleaner和__init__
            - nonen_clean_text_inf 针对混合语种{"zh", "ja","auto"}
                - splite_en_inf
        """
        if language in {"en","all_zh","all_ja"}:
            phones, word2ph, norm_text = clean_text_inf(text, language)
        elif language in {"zh", "ja","auto"}:
            phones, word2ph, norm_text = nonen_clean_text_inf(text, language)
        return phones, word2ph, norm_text
    
    def get_bert_inf(self, phones, word2ph, norm_text, language):
        device = self.device # 【补】
        is_half = self.is_half # 【补】
        
        language=language.replace("all_","")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)

        return bert
    
    def nonen_get_bert_inf(self, text, language):
        if(language!="auto"):
            textlist, langlist = splite_en_inf(text, language)
        else:
            textlist=[]
            langlist=[]
            for tmp in LangSegment.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        bert_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)

        return bert
    
    def get_bert_feature(self, text, word2ph):

        is_half = self.is_half # 【补】
        device = self.device # 【补】

        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)  #####输入是long不用管精度问题，精度随bert_model
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        if(is_half==True):phone_level_feature=phone_level_feature.half()
        
        return phone_level_feature.T

    def get_bert_final(self,phones, word2ph, text,language):
        """
        根据语言 选择调用不同的函数来得到一个bert表示。
        需要输入Get_clean_text_final得到的文字素材
        -> bert
            - get_bert_inf 针对纯英文”en”
            - nonen_get_bert_inf 针对混合语种{"zh", "ja","auto"}
            - get_bert_feature 针对纯中文”all_zh”
        """
        device = self.device # 【补】

        if language == "en":
            bert = self.get_bert_inf(phones, word2ph, text, language) # 【补】
        elif language in {"zh", "ja","auto"}:
            bert = self.nonen_get_bert_inf(text, language)
        elif language == "all_zh":
            bert = self.get_bert_feature(text, word2ph).to(device)
        else:
            bert = torch.zeros((1024, len(phones))).to(device)
        return bert

    # ===
    # ======适配混合语种输出======

    def get_tts_wav(self, text, text_language="zh", wav_save_path="temp/output.wav",
                    how_to_cut="凑四句一切", 
                    top_k=20, top_p=0.6, temperature=0.6, 
                    # 关于上面三个参数 https://github.com/RVC-Boss/GPT-SoVITS/pull/457
                    # 可以通过降低温度，降低top_p,top_k 提升模型输出内容的一致性
                    ref_free = False): # 在不知道参考音频文本的情况下进行推理
        
        # ====== 函数内变量 ======
        # ===
        # 根据声色指定相关模型与参考语音
        ref_wav_path = self.ref_wav_path

        # 针对sunshine_girl模型在refer_free = False时出现吞字情况
        # 强行指定其refer_free = True
        if self.audio_model_id in ["X-D-Lab/TTS-GPT_SoVITS-sunshine_girl","/X-D-Lab/TTS-GPT_SoVITS-sunshine_girl"]:
            ref_free = True

        if not ref_free:
            prompt_text_path = self.prompt_text_path
            with open(prompt_text_path, 'r', encoding='utf-8') as file:
                prompt_text = file.read()
            # 如果txt中音频文本为空，则也不使用音频文本。
            if prompt_text is None or len(prompt_text) == 0:
                ref_free = True
        prompt_language = self.prompt_language

        wav_save_path = "%s/%s"%(root_dir,wav_save_path)
        
        device = self.device
        is_half = self.is_half
        dtype = self.dtype

        hz = 50
        max_sec = self.config['data']['max_sec']
        # ===
        # ====== 函数内变量 ======


        # 确认参考语音和推理文本的语种(可以不必，已对prompt_language和text_language的输入做了严格限制)
        prompt_language = dict_language[prompt_language]
        text_language = dict_language[text_language]

        if not ref_free:
            prompt_text = prompt_text.strip("\n")
            if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
            #【日志】 print("实际输入的参考文本:", prompt_text)
       
        # 预处理推理文本：文本第一段(get_first)若特别短<4字符，则在文本最前方加上句号。
        text = text.strip("\n")
        if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text
        
        #【日志】 print("实际输入的目标文本:", text)

        # 创建空音频段
        # 第一个with torch.no_grad() 从参考音频中提取语义信息,并把空音频段放到参考音频末尾->prompt_semantic
        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * 0.3), # 【补】
            dtype=np.float16 if is_half == True else np.float32,
        )
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise OSError("参考音频在3~10秒范围外，请更换！")
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if is_half == True:
                wav16k = wav16k.half().to(device)
                zero_wav_torch = zero_wav_torch.half().to(device)
            else:
                wav16k = wav16k.to(device)
                zero_wav_torch = zero_wav_torch.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = self.vq_model.extract_latent(ssl_content)
    
            prompt_semantic = codes[0, 0]

        # 切分推理文本，5种方法。一般可选4句一切和按标点符号切。之后，将其中小于5的语句/短语合并(merge_short_text_in_array)。最终得到推理文本切割列表
        # -> texts
        if (how_to_cut == "凑四句一切"):
            text = cut1(text)
        elif (how_to_cut == "凑50字一切"):
            text = cut2(text)
        elif (how_to_cut == "按中文句号。切"):
            text = cut3(text)
        elif (how_to_cut == "按英文句号.切"):
            text = cut4(text)
        elif (how_to_cut == "按标点符号切"):
            text = cut5(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
            
        #【日志】 print("实际输入的目标文本(切句后):", text)
        texts = text.split("\n")
        texts = merge_short_text_in_array(texts, 5)
        audio_opt = []
        if not ref_free:
            # 处理参考文本(get_cleaned_text_final)得到文字素材
            # -> phones1,word2ph1,norm_text1
            phones1, word2ph1, norm_text1=self.get_cleaned_text_final(prompt_text, prompt_language)
            # 处理参考语音(Get_bert_final) 输入文字素材phones1,word2ph1,norm_text1
            # 得到bert表示
            # ->bert1
            bert1=self.get_bert_final(phones1, word2ph1, norm_text1,prompt_language).to(dtype)

        # for循环 处理推理文本,对texts中的每一段语句/短语
        # 处理文本(get_cleaned_text_final)得到文字素材
        # -> phones2,word2ph2,norm_text2
        # 处理参考语音(Get_bert_final) 输入文字素材phones2,word2ph2,norm_text2
        # 得到bert表示
        # ->bert2
        for text in texts:
            # 解决输入目标文本的空行导致报错的问题
            if (len(text.strip()) == 0):
                continue
            if (text[-1] not in splits): text += "。" if text_language != "en" else "."
            # 【日志】print("实际输入的目标文本(每句):", text)
            phones2, word2ph2, norm_text2 = self.get_cleaned_text_final(text, text_language)
            bert2 = self.get_bert_final(phones2, word2ph2, norm_text2, text_language).to(dtype)
            if not ref_free:
                bert = torch.cat([bert1, bert2], 1)
                all_phoneme_ids = torch.LongTensor(phones1+phones2).to(device).unsqueeze(0)
            else:
                bert = bert2
                all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

            bert = bert.to(device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
            prompt = prompt_semantic.unsqueeze(0).to(device)
            
            with torch.no_grad():
                # pred_semantic = t2s_model.model.infer(
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=hz * max_sec,
                )
            
            # print(pred_semantic.shape,idx)
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(
                0
            )  # .unsqueeze(0)#mq要多unsqueeze一次
            refer = get_spepc(self.hps, ref_wav_path)  # .to(device)  # 【补】
            if is_half == True:
                refer = refer.half().to(device)
            else:
                refer = refer.to(device)
            # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
            audio = (
                self.vq_model.decode( # 【补】
                    pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
                )
                    .detach()
                    .cpu()
                    .numpy()[0, 0]
            )  ###试试重建不带上prompt部分
            max_audio=np.abs(audio).max()#简单防止16bit爆音
            if max_audio>1:audio/=max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
            
        sampling_rate, audio_data = self.hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
            np.int16
        )

        sf.write(wav_save_path, audio_data, sampling_rate, format='wav')
        torch.cuda.empty_cache()
