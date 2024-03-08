"""
# api.py usage

python api_alpha.py

` python api_alpha.py -l ja`

## 执行参数:
`-l` - `默认参考音频语种默认zh, "中文","英文","日文","zh","en","ja"`

`-d` - `推理设备, "cuda","cpu","mps"`
`-a` - `绑定地址, 默认"0.0.0.0"`
`-p` - `绑定端口, 默认9880, 可在 config.py 中指定`
`-fp` - `覆盖 config.py 使用全精度`
`-hp` - `覆盖 config.py 使用半精度`

`-hb` - `cnhubert路径`
`-b` - `bert路径`

## 调用:

### 推理

endpoint: `/`

使用执行参数指定的参考音频:
GET:
    `http://0.0.0.0:9880?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh&role=大格蕾修`
POST:
```json
{
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh",
    "role":"大格蕾修"
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400


### 刷新role_configs（包括模型和参考音频）

endpoint: `/refresh_role_configs`

GET:
    `http://0.0.0.0:9880/refresh_role_configs`


RESP:
成功: json, http code 200
失败: json, 400

### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
    `http://0.0.0.0:9880/control?command=restart`
POST:
```json
{
    "command": "restart"
}
```

RESP: 无

"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import argparse
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import signal
from time import time as ttime
import torch
import librosa
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException, Depends, status,Header
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from feature_extractor import cnhubert
from io import BytesIO
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
import config_alpha as global_config

g_config = global_config.Config()

# AVAILABLE_COMPUTE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="GPT-SoVITS api")


parser.add_argument("-l", "--refer_language", type=str, default="zh", help="默认参考音频语种")

parser.add_argument("-d", "--device", type=str, default=g_config.infer_device, help="cuda / cpu / mps")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 0.0.0.0")
parser.add_argument("-p", "--port", type=int, default=g_config.api_port, help="default: 9880")
parser.add_argument("-fp", "--full_precision", action="store_true", default=False, help="覆盖config.is_half为False, 使用全精度")
parser.add_argument("-hp", "--half_precision", action="store_true", default=False, help="覆盖config.is_half为True, 使用半精度")
# bool值的用法为 `python ./api.py -fp ...`
# 此时 full_precision==True, half_precision==False

parser.add_argument("-hb", "--hubert_path", type=str, default=g_config.cnhubert_path, help="覆盖config.cnhubert_path")
parser.add_argument("-b", "--bert_path", type=str, default=g_config.bert_path, help="覆盖config.bert_path")

args = parser.parse_args()

device = args.device
port = args.port
host = args.bind_addr
refer_language=args.refer_language

is_half = g_config.is_half
if args.full_precision:
    is_half = False
if args.half_precision:
    is_half = True
if args.full_precision and args.half_precision:
    is_half = g_config.is_half  # 炒饭fallback

print(f"[INFO] 半精: {is_half}")

cnhubert_base_path = args.hubert_path
bert_path = args.bert_path

cnhubert.cnhubert_base_path = cnhubert_base_path
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def is_empty(*items):  # 任意一项不为空返回False
    for item in items:
        if item is not None and item != "":
            return False
    return True


def is_full(*items):  # 任意一项为空返回False
    for item in items:
        if item is None or item == "":
            return False
    return True


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  #####输入是long不用管精度问题，精度随bert_model
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # if(is_half==True):phone_level_feature=phone_level_feature.half()
    return phone_level_feature.T


class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)

import json
import subprocess
def generate_role_configs(base_path='role_models'):
    role_configs = {}
    # 遍历base_path下的所有文件夹和文件
    for role in os.listdir(base_path):
        role_path = os.path.join(base_path, role)
        if os.path.isdir(role_path):
            role_configs[role] = {}
            for file in os.listdir(role_path):
                if file.endswith('.pth'):
                    role_configs[role]['sovits_path'] = os.path.join(role_path, file)
                elif file.endswith('.ckpt'):
                    role_configs[role]['gpt_path'] = os.path.join(role_path, file)
                elif file.endswith('.wav'):
                    role_configs[role]['refer_audio_path'] = os.path.join(role_path, file)
                else:
                    # 检查文件是否为支持的音频格式，且不是wav
                    audio_exts = ['.mp3', '.ogg', '.flac', '.aac']
                    file_ext = os.path.splitext(file)[1]
                    if file_ext in audio_exts:
                        original_path = os.path.join(role_path, file)
                        wav_path = original_path.rsplit('.', 1)[0] + '.wav'
                        # 使用ffmpeg转换音频到wav
                        cmd = ['ffmpeg', '-i', original_path, wav_path]
                        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        # 更新配置路径为新的wav文件
                        role_configs[role]['refer_audio_path'] = wav_path
                        # 删除原始文件
                        os.remove(original_path)
    # 将字典保存为JSON文件
    with open('role_configs.json', 'w', encoding='utf-8') as f:
        json.dump(role_configs, f, ensure_ascii=False, indent=4)
    
    return role_configs

# 调用函数生成字典并保存为JSON
global role_configs
role_configs = generate_role_configs() #程序启动时更新role_configs
print(role_configs)

class RoleModelManager:
    def __init__(self, role_config):
        self.role_config = role_config
        self.models_cache_vq = {}
        self.models_cache_t2s = {}
        self.value_cache_hps = {}
        self.value_cache_ssl_model = {}
        self.value_cache_config = {}

    def load_model(self, role_name, device):
        if role_name in self.models_cache_vq:
            return self.models_cache_vq[role_name],self.models_cache_t2s[role_name],self.value_cache_hps[role_name],self.value_cache_ssl_model[role_name],self.value_cache_config[role_name]
        
        if role_name not in self.role_config:
            raise ValueError(f"角色{role_name}的配置不存在。")  # Added the missing expression in the error message
        
        role_config = self.role_config[role_name]
        # 加载模型和配置
        sovits_path=role_config["sovits_path"]
        gpt_path=role_config["gpt_path"]
        n_semantic = 1024
        dict_s2 = torch.load(sovits_path, map_location="cpu")
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        dict_s1 = torch.load(gpt_path, map_location="cpu")
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
        # 从这里加入缓存
        print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        if is_half:
            t2s_model = t2s_model.half()
        t2s_model = t2s_model.to(device)
        t2s_model.eval()
        total = sum([param.nelement() for param in t2s_model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
                
        self.models_cache_vq[role_name] = vq_model
        self.models_cache_t2s[role_name] = t2s_model
        self.value_cache_hps[role_name] = hps
        self.value_cache_ssl_model[role_name] = ssl_model
        self.value_cache_config[role_name] = config

        return vq_model,t2s_model,hps,ssl_model,config

# 初始化模型管理器
global model_manager
model_manager = RoleModelManager(role_configs)


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                             hps.data.win_length, center=False)
    return spec


dict_language = {
    "中文": "zh",
    "英文": "en",
    "日文": "ja",
    "ZH": "zh",
    "EN": "en",
    "JA": "ja",
    "zh": "zh",
    "en": "en",
    "ja": "ja"
}


import re
def split_texts_by_punctuation(texts):
    # 定义中英文标点符号集合
    punctuations = r'[,.!?;:""<>（）《》【】、|！？；：“”‘’，。]'
    # 使用正则表达式按标点符号切分文本，同时保留标点符号
    split_texts = re.split(f'({punctuations})', texts)
    # 过滤空字符串
    split_texts = [text for text in split_texts if text not in ['', ' ', '\n']]
    
    # 处理首位标点符号：如果第一个元素是标点符号，则丢弃，比如：'“你好”' -> '你好”'，为了适配阅读软件的规则
    if split_texts and re.match(punctuations, split_texts[0]):
        split_texts = split_texts[1:]

    # 将标点符号和前面的文本合并
    combined_texts = []
    for i in range(len(split_texts)):
        if i > 0 and re.match(punctuations, split_texts[i]):
            combined_texts[-1] += split_texts[i]  # 将标点符号添加到前一个文本元素
        else:
            combined_texts.append(split_texts[i])  # 否则，作为新的文本元素添加

    return combined_texts



def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language,role):
    vq_model,t2s_model,hps,ssl_model,config = model_manager.load_model(role, device)
    hz = 50
    max_sec = config['data']['max_sec']
    t0 = ttime()
    prompt_text = prompt_text.strip("\n")
    prompt_language, text = prompt_language, text.strip("\n")
    zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if is_half == True else np.float32)

    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if (is_half == True):
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
    t1 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    phones1, word2ph1, norm_text1 = clean_text(prompt_text, prompt_language)
    phones1 = cleaned_text_to_sequence(phones1)
    #texts = text.split("\n")
    texts = split_texts_by_punctuation(text) # 按中英文标点符号切分文本
    audio_opt = []

    for text in texts:
        phones2, word2ph2, norm_text2 = clean_text(text, text_language)
        phones2 = cleaned_text_to_sequence(phones2)
        if (prompt_language == "zh"):
            bert1 = get_bert_feature(norm_text1, word2ph1).to(device)
        else:
            bert1 = torch.zeros((1024, len(phones1)), dtype=torch.float16 if is_half == True else torch.float32).to(
                device)
        if (text_language == "zh"):
            bert2 = get_bert_feature(norm_text2, word2ph2).to(device)
        else:
            bert2 = torch.zeros((1024, len(phones2))).to(bert1)
        bert = torch.cat([bert1, bert2], 1)

        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = prompt_semantic.unsqueeze(0).to(device)
        t2 = ttime()
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=config['inference']['top_k'],
                early_stop_num=hz * max_sec)
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path)  # .to(device)
        if (is_half == True):
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = \
            vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0),
                            refer).detach().cpu().numpy()[
                0, 0]  ###试试重建不带上prompt部分
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)


def handle_control(command):
    if command == "restart":
        os.execl(g_config.python_exec, g_config.python_exec, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)





def handle2(text, text_language,role):
    if is_empty(role_configs):
        raise HTTPException(status_code=400, detail="role_configs为空,请检查配置文件")
    if role not in role_configs:
        raise HTTPException(status_code=400, detail="在role_configs中没有找到角色:{}".format(role))
    refer_wav_path=role_configs[role]["refer_audio_path"]
    prompt_text=role_configs[role]["refer_audio_path"].split('/')[-1].split('.')[0]
    prompt_language=refer_language

    if is_empty(refer_wav_path, prompt_text):
        raise HTTPException(status_code=400, detail="role_configs中的refer_audio_path为空,请检查配置文件")

    with torch.no_grad():
        gen = get_tts_wav(
            refer_wav_path, prompt_text, prompt_language, text, text_language,role
        )
        sampling_rate, audio_data = next(gen)

    wav = BytesIO()
    sf.write(wav, audio_data, sampling_rate, format="wav")
    wav.seek(0)

    torch.cuda.empty_cache()
    if device == "mps":
        print('executed torch.mps.empty_cache()')
        torch.mps.empty_cache()
    return StreamingResponse(wav, media_type="audio/wav")

# 添加一个依赖项函数用于API密钥验证
def api_key_auth(x_api_key: str = Header(None)):
    if g_config.ENABLE_AUTH and x_api_key != g_config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key"
        )

app = FastAPI()


@app.post("/control")
async def control(request: Request, _=Depends(api_key_auth)):
    json_post_raw = await request.json()
    return handle_control(json_post_raw.get("command"))


@app.get("/control")
async def control(command: str = None, _=Depends(api_key_auth)):
    return handle_control(command)


@app.post("/")
async def tts_endpoint(request: Request, _=Depends(api_key_auth)):
    json_post_raw = await request.json()
    return handle2(
        json_post_raw.get("text"),
        json_post_raw.get("text_language"),
        json_post_raw.get("role")
    )


@app.get("/")
async def tts_endpoint(
        text: str = None,
        text_language: str = None,
        role: str = None,
        _=Depends(api_key_auth)
):
    return handle2(text, text_language, role)

#添加接口，刷新role_configs，并向前端返回role_configs
@app.get("/refresh_role_configs")
async def refresh_role_configs(_=Depends(api_key_auth)):
    global role_configs
    global model_manager
    role_configs = generate_role_configs()
    model_manager = RoleModelManager(role_configs)
    return JSONResponse(content=role_configs)


if __name__ == "__main__":
    uvicorn.run("api_alpha:app", host=host, port=port, workers=3, reload=True)  # reload=True表示修改代码后自动重启
    #reload 选项可以检测到整个 py 文件中的修改，以及任何其他被该文件直接或间接导入的 Python 文件的修改