import sys,os

import torch

# 推理用的指定模型放入role_models文件夹,以角色名命名子文件夹，文件夹内放入模型文件和参考音频文件（以字幕内容作为文件名）
# 例如：role_models/八重神子/role_models/经典的青春友情剧，不管怎么样都百看不厌啊。.wav
is_half = eval(os.environ.get("is_half","True"))
is_share=False

# 默认不启用身份验证
ENABLE_AUTH = False  # 默认不启用身份验证
API_KEY = "xhwang"  # 将此设置为您的API密钥

cnhubert_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"

exp_root = "logs"
python_exec = sys.executable or "python"
if torch.cuda.is_available():
    infer_device = "cuda"
elif torch.backends.mps.is_available():
    infer_device = "mps"
else:
    infer_device = "cpu"

webui_port_main = 9874
webui_port_uvr5 = 9873
webui_port_infer_tts = 9872
webui_port_subfix = 9871

api_port = 9880

if infer_device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    if (
            ("16" in gpu_name and "V100" not in gpu_name.upper())
            or "P40" in gpu_name.upper()
            or "P10" in gpu_name.upper()
            or "1060" in gpu_name
            or "1070" in gpu_name
            or "1080" in gpu_name
    ):
        is_half=False

if(infer_device=="cpu"):is_half=False

class Config:
    def __init__(self):
        self.is_half = is_half
        self.ENABLE_AUTH = ENABLE_AUTH
        self.API_KEY = API_KEY
        self.cnhubert_path = cnhubert_path
        self.bert_path = bert_path
        self.exp_root = exp_root
        self.python_exec = python_exec
        self.infer_device = infer_device

        self.webui_port_main = webui_port_main
        self.webui_port_uvr5 = webui_port_uvr5
        self.webui_port_infer_tts = webui_port_infer_tts
        self.webui_port_subfix = webui_port_subfix

        self.api_port = api_port
