import os
import json
import torch
import asyncio
import librosa
import hashlib
import edge_tts
import gradio as gr
from config import Config
from vc_infer_pipeline import VC
from fairseq import checkpoint_utils
from lib.infer_pack.models import (SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono, SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono,)

config = Config()

def load_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f: content = json.load(f)
    return content

def file_checksum(file_path):
    with open(file_path, 'rb') as f:
        file_data = f.read()
        return hashlib.md5(file_data).hexdigest()

def get_existing_model_info(category_directory):
    model_info_path = os.path.join(category_directory, 'model_info.json')
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f: return json.load(f)
    return None

def generate_model_info_files():
    folder_info = {}
    model_directory = "models/"
    for category_name in os.listdir(model_directory):
        category_directory = os.path.join(model_directory, category_name)
        if not os.path.isdir(category_directory): continue

        folder_info[category_name] = {"title": category_name, "folder_path": category_name}
        existing_model_info = get_existing_model_info(category_directory)
        model_info = {}
        regenerate_model_info = False

        for model_name in os.listdir(category_directory):
            model_path = os.path.join(category_directory, model_name)
            if not os.path.isdir(model_path): continue

            model_data, regenerate = gather_model_info(category_directory, model_name, model_path, existing_model_info)
            if model_data is not None:
                model_info[model_name] = model_data
                regenerate_model_info |= regenerate

        if regenerate_model_info:
            with open(os.path.join(category_directory, 'model_info.json'), 'w') as f: json.dump(model_info, f, indent=4)

    folder_info_path = os.path.join(model_directory, 'folder_info.json')
    with open(folder_info_path, 'w') as f: json.dump(folder_info, f, indent=4)

def should_regenerate_model_info(existing_model_info, model_name, pth_checksum, index_checksum):
    if existing_model_info is None or model_name not in existing_model_info: return True
    return (existing_model_info[model_name]['model_path_checksum'] != pth_checksum or existing_model_info[model_name]['index_path_checksum'] != index_checksum)

def get_model_files(model_path): return [f for f in os.listdir(model_path) if f.endswith('.pth') or f.endswith('.index')]

def gather_model_info(category_directory, model_name, model_path, existing_model_info):
    model_files = get_model_files(model_path)
    if len(model_files) != 2: return None, False

    pth_file = [f for f in model_files if f.endswith('.pth')][0]
    index_file = [f for f in model_files if f.endswith('.index')][0]
    pth_checksum = file_checksum(os.path.join(model_path, pth_file))
    index_checksum = file_checksum(os.path.join(model_path, index_file))
    regenerate = should_regenerate_model_info(existing_model_info, model_name, pth_checksum, index_checksum)

    return {"title": model_name, "model_path": pth_file, "feature_retrieval_library": index_file, "model_path_checksum": pth_checksum, "index_path_checksum": index_checksum}, regenerate

def create_vc_fn(model_name, tgt_sr, net_g, vc, if_f0, version, file_index):
    def vc_fn(tts_text, tts_voice):
        try:
            if len(tts_text) > 100: return None
            if tts_text is None or tts_voice is None: return None
            asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save("tts.mp3"))
            audio, sr = librosa.load("tts.mp3", sr=16000, mono=True)
            vc_input = "tts.mp3"
            times = [0, 0, 0]
            audio_opt = vc.pipeline(hubert_model, net_g, 0, audio, vc_input, times, 0, "pm", file_index, 0.7, if_f0, 3, tgt_sr, 0, 1, version, 0.5, f0_file=None)
            return (tgt_sr, audio_opt)
        except Exception: return None
    return vc_fn

def load_model_parameters(category_folder, character_name, info):
    model_index = f"models/{category_folder}/{character_name}/{info['feature_retrieval_library']}"
    cpt = torch.load(f"models/{category_folder}/{character_name}/{info['model_path']}", map_location="cpu")
    return model_index, cpt

def select_net_g(cpt, version, if_f0):
    if version == "v1":
        if if_f0 == 1: net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:  net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1: net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else: net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    return net_g

def load_model_and_prepare(cpt, net_g):
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(config.device)
    net_g = net_g.half() if config.is_half else net_g.float()
    return net_g

def create_and_append_model(models, model_functions, character_name, model_title, version, vc_fn):
    models.append((character_name, model_title, version, vc_fn))
    model_functions[character_name] = vc_fn
    return models, model_functions

def load_model():
    categories = []
    model_functions = {}
    folder_info = load_json_file("models/folder_info.json")
    for category_name, category_info in folder_info.items():
        models = []
        models_info = load_json_file(f"models/{category_info['folder_path']}/model_info.json")
        for character_name, info in models_info.items():
            model_index, cpt = load_model_parameters(category_info['folder_path'], character_name, info)
            net_g = select_net_g(cpt, cpt.get("version", "v1"), cpt.get("f0", 1))
            cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
            net_g = load_model_and_prepare(cpt, net_g)
            vc = VC(cpt["config"][-1], config)
            vc_fn = create_vc_fn(info['model_path'], cpt["config"][-1], net_g, vc, cpt.get("f0", 1), cpt.get("version", "v1"), model_index)
            models, model_functions = create_and_append_model(models, model_functions, character_name, info['title'], cpt.get("version", "v1"), vc_fn)
        categories.append([category_info['title'], category_info['folder_path'], models])
    return categories, model_functions

generate_model_info_files()

css = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
footer { visibility: hidden; display: none; }
.center-container { display: flex; flex-direction: column; align-items: center; justify-content: center;}
"""

if __name__ == '__main__':
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"], suffix="")
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    hubert_model = hubert_model.half() if config.is_half else hubert_model.float()
    hubert_model.eval()
    categories, model_functions = load_model()
    tts_voice_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())
    voices = [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]
    with gr.Blocks(css=css, title="Demo RVC TTS - Pavloh", theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="blue", radius_size="lg", text_size="lg")
                   .set(loader_color="#0B0F19", shadow_drop='*shadow_drop_lg', block_border_width="3px")) as pavloh:
        gr.HTML("""
            <div class="center-container">
                <div style="display: flex; justify-content: center;">
                    <a href="https://github.com/ImPavloh/rvc-tts/blob/main/LICENSE" target="_blank">
                        <img src="https://img.shields.io/github/license/impavloh/voiceit?style=for-the-badge&logo=github&logoColor=white" alt="License">
                    </a>
                    <a href="https://github.com/ImPavloh/rvc-tts" target="_blank">
                        <img src="https://img.shields.io/badge/repository-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
                    </a>
                    <form action="https://www.paypal.com/donate" method="post" target="_blank">
                        <input type="hidden" name="hosted_button_id" value="6FPWP9AWEKSWJ" />
                        <input type="image" src="https://img.shields.io/badge/support-%2300457C.svg?style=for-the-badge&logo=paypal&logoColor=white" border="0" name="submit" alt="Donate with PayPal" />
                        <img alt="" border="0" src="https://www.paypal.com/es_ES/i/scr/pixel.gif" width="1" height="1" />
                    </form>
                    <a href="https://twitter.com/impavloh" target="_blank">
                        <img src="https://img.shields.io/badge/follow-%231DA1F2.svg?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter">
                    </a>
                </div>
                <div style="display: inline-flex; align-items: center; gap: 0.8rem; font-size: 1.75rem;">
                    <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px">🗣️ RVC TTS Demo - <a style="text-decoration: underline;" href="https://twitter.com/impavloh">Pavloh</a></h1>
                </div>
                <p style="margin-bottom: 10px; font-size: 94%; line-height: 23px;">An AI-Powered Text-to-Speech</p>
                <p><b>Try out the <a style="text-decoration: underline;" href="https://github.com/ImPavloh/rvc-tts-discord-bot">RVC Text-to-Speech Discord Bot</a> for yourself!</b></p>
            </div>
        """)

        with gr.Row():
            with gr.Column():
                m1 = gr.Dropdown(label="📦 Voice Model", choices=list(model_functions.keys()), allow_custom_value=False, value="Ibai")
                t1 = gr.Textbox(label="📝 Text to convert")
                t2 = gr.Dropdown(label="⚙️ Voice style and language [Edge-TTS]", choices=voices, allow_custom_value=False, value="es-ES-AlvaroNeural-Male")
                c1 = gr.Button("Convert", variant="primary")
                a1 = gr.Audio(label="🔉 Converted Text", interactive=False)

                def call_selected_model_fn(selected_model, t1, t2):
                    vc_fn = model_functions[selected_model]
                    return vc_fn(t1, t2)

                c1.click(fn=call_selected_model_fn, inputs=[m1, t1, t2], outputs=[a1])

                gr.HTML("""
                        <center>
                            <p><i> By using this website, you agree to the <a style="text-decoration: underline;" href="https://github.com/ImPavloh/rvc-tts-discord-bot/blob/main/LICENSE">license</a>. </i></p>
                        </center>
                    """)

pavloh.queue(concurrency_count=1).launch()
