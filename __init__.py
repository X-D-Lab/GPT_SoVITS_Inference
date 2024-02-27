import sys,os
#gpt_sovits_tts的根目录 ，每次从gpt_sovits_tts导入模块时执行，确保路径不乱飘
root_dir = os.path.dirname(__file__)
sys.path.append(root_dir)
sys.path.append("%s/GPT_SoVITS"%(root_dir))