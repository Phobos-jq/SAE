import sys
sys.path.append("../")
import torch
import tiktoken
import os
import pickle
from contextlib import nullcontext
from nanoGPT.model import GPTConfig, GPT
import json
import blobfile as bf
from train import FastAutoencoder, make_torch_comms
# os.environ["CUDA_VISIBLE_DEVICES"]="6"
model_dir = '/data4/jqliu/ML_jq/nanoGPT/out_ori/out_test'
sae_dir = '/data4/jqliu/ML_jq/SAE/save/ori_8192_128'

N_DIR = 8192
K = 128

init_from = 'resume' # either 'resume' (from an model_dir) or a gpt2 variant (e.g. 'gpt2-xl')
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device_gpt = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
exec(open('/home/jqliu/ML_jq/git_non-neg_interpretability/nanoGPT/configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device_gpt else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(model_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device_gpt)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device_gpt)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
# ====================模型加载完成========================




# Extract neuron activations with transformer_lens
device = next(model.parameters()).device

prompt = "This is an example of a prompt that."
tokens = torch.tensor(encode(prompt)).unsqueeze(0)  # (1, n_tokens)
with torch.no_grad():
    logits, loss, activation_cache = model.run_with_cache(tokens)
    
class Config:
    n_op_shards: int = 1
    n_replicas: int = 1

    n_dirs: int = N_DIR
    bs: int = 131072
    d_model: int = 768
    k: int = K
    auxk: int = 256

    lr: float = 1e-4
    eps: float = 6.25e-10
    clip_grad: float | None = None
    auxk_coef: float = 1 / 32
    dead_toks_threshold: int = 10_000_000
    ema_multiplier: float | None = None
    
cfg = Config()
comms = make_torch_comms(n_op_shards=cfg.n_op_shards, n_replicas=cfg.n_replicas)
n_dirs_local = cfg.n_dirs // cfg.n_op_shards
bs_local = cfg.bs // cfg.n_replicas
config ={
    "n_dirs_local": n_dirs_local,
    "d_model": cfg.d_model,
    "k": cfg.k,
    "auxk": cfg.auxk,
    "dead_steps_threshold": cfg.dead_toks_threshold // cfg.bs,
    "comms": comms,
}
with bf.BlobFile(sae_dir, mode="rb") as f:
    state_dict = torch.load(f)
    autoencoder = FastAutoencoder.from_state_dict(state_dict, **config)
    autoencoder.to(device)

input_tensor = activation_cache['features'].to('cuda').reshape(-1,768)

input_tensor_ln = input_tensor

with torch.no_grad():
    encode_dict = autoencoder.encode(input_tensor_ln)
    """
    encode_dict (dict): 包含 'inds' 和 'vals' 的字典，'inds' 和 'vals' 均为形状 (batch_size, seq_len, k) 的张量。
    """
    reconstructed_activations = autoencoder.decode_sparse(**encode_dict)

normalized_mse = (reconstructed_activations - input_tensor).pow(2).sum(dim=1) / (input_tensor).pow(2).sum(dim=1)
print(normalized_mse)