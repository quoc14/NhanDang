import sys

from transformers import AutoModel
from huggingface_hub import hf_hub_download
import shutil
import os
import sys

list_all_models = ["minchul/cvlface_DFA_mobilenet", 
                   "minchul/cvlface_DFA_resnet50",
                   "minchul/cvlface_adaface_vit_base_webface4m",
                   "minchul/cvlface_DFA_resnet50",
                   "minchul/cvlface_adaface_ir18_vgg2",
                   "minchul/cvlface_adaface_ir18_webface4m",
                   "minchul/cvlface_adaface_ir50_webface4m",
                   "minchul/cvlface_adaface_ir50_casia",
                   "minchul/cvlface_adaface_ir50_ms1mv2",
                   "minchul/cvlface_adaface_ir101_ms1mv2",
                   "minchul/cvlface_adaface_ir101_ms1mv3",
                   "minchul/cvlface_adaface_ir101_webface4m",
                   "minchul/cvlface_adaface_vit_base_kprpe_webface12m",
                   "minchul/cvlface_adaface_ir101_webface12m",
                   "minchul/cvlface_adaface_vit_base_webface4m",
                   "minchul/cvlface_adaface_vit_base_kprpe_webface4m"
                   ]

# helpfer function to download huggingface repo and use model
def download(repo_id, path, HF_TOKEN=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)
    with open(os.path.join(path, 'files.txt'), 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)

def download_all_models():
    for model in list_all_models:
        print("-----------------Downloading model: ", model, "-----------------")
        download(model, os.path.abspath(f"model/{model}"))


# helpfer function to download huggingface repo and use modelcd 
def load_model_from_local_path(path, HF_TOKEN=None):
    path = os.path.abspath(path)
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, token=HF_TOKEN)
    os.chdir(cwd)
    sys.path.pop(0)
    return model


# helpfer function to download huggingface repo and use model
def load_model_by_repo_id(repo_id, save_path, HF_TOKEN=None, force_download=False):
    if force_download:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)