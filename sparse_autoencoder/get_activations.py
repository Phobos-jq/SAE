"""
To get the max activations of the features
"""
import sys
sys.path.append("/home/jqliu/ML_jq/git_non-neg_interpretability/")
sys.path.append("/home/jqliu/ML_jq/sparse_autoencoder")
import os
import signal
from tqdm import tqdm
import numpy as np
import tiktoken
import datasets as d  # huggingface datasets
import pickle
from contextlib import nullcontext
import torch
from nanoGPT.model import GPTConfig, GPT
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
import gc
import json
import blobfile as bf
from sparse_autoencoder.train import FastAutoencoder, make_torch_comms, load_activations, init_from_data_
import random
random.seed(42)
os.environ["CUDA_VISIBLE_DEVICES"]="7"

N_DIR = 8192
K = 128
max_length = 64
batch_size = 512
MAX_BATCH_NUM = 64
NO_SAME_MAXTOKEN = False
MEAN_ACT = False # True: get the most activated sentence(measured by mean activations)
save_file_path = '/data4/jqliu/ML_jq/SAE/save'
folder_name = '8192_128_nonneg+'
save_file_name = 'features_activations'  # '.../folder_name/features_activations__len_{max_length}__shard_{shard_index}.json'

sae_dir = '/data4/jqliu/ML_jq/SAE/save/nonneg+_8192_128'
model_dir = '/data4/jqliu/ML_jq/nanoGPT/out_feature+weight/out_test' # ignored if init_from is not 'resume'
enc = tiktoken.get_encoding("gpt2")

folder_path = os.path.join(save_file_path, folder_name)
if not os.path.exists(folder_path):
    print("Making", folder_path)
    os.mkdir(folder_path)
print("Now geting activations in features layer")
print(f"{max_length=}, {folder_path=}, {model_dir=}, {MEAN_ACT=}, {NO_SAME_MAXTOKEN=}")

max_tokens = [[] for _ in range(768)]
selected_index = [random.randint(0, N_DIR) for _ in range(768)]
with open(folder_path+'/selected_index.json', 'w') as f:
    json.dump(selected_index, f, indent=4)

# 定义信号处理函数
def signal_handler(sig, frame):
    print("捕获到中断信号，正在清理资源...")
    # 在此添加任何清理代码，例如关闭文件、释放资源等
    if 'executor' in globals():
        executor.shutdown(wait=True)
    # 如果有必要，执行其他清理操作
    gc.collect()
    print("资源清理完毕，程序退出。")
    exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
# 假设 activations 是激活值字典
def convert_activations_to_json(activations):
    json_ready_activations = {}

    for neuron_idx in range(768):
            json_ready_activations[f"feature_{neuron_idx}"] = {
                "max_activation_value": (
                    activations[f"feature_{neuron_idx}"]["max_activation_value"].item() 
                    if isinstance(activations[f"feature_{neuron_idx}"]["max_activation_value"], (torch.Tensor, np.generic)) 
                    else activations[f"feature_{neuron_idx}"]["max_activation_value"]
                ),
                "max_token_idx": (
                    int(activations[f"feature_{neuron_idx}"]["max_token_idx"].item()) 
                    if isinstance(activations[f"feature_{neuron_idx}"]["max_token_idx"], (torch.Tensor, np.generic)) 
                    else int(activations[f"feature_{neuron_idx}"]["max_token_idx"])
                ),
                "max_example_index": int(activations[f"feature_{neuron_idx}"]["max_example_index"]),
                "max_token_index": int(activations[f"feature_{neuron_idx}"]["max_token_index"]),
                "activations": activations[f"feature_{neuron_idx}"]["activations"].tolist(),
                "mean_activation": (
                    activations[f"feature_{neuron_idx}"]["mean_activation"].item() 
                    if isinstance(activations[f"feature_{neuron_idx}"]["mean_activation"], (torch.Tensor, np.generic)) 
                    else activations[f"feature_{neuron_idx}"]["mean_activation"]
                ),
                "activations_first_sample": activations[f"feature_{neuron_idx}"]["activations_first_sample"].tolist()
            }
    return json_ready_activations

def create_batches(tokenized_shard, batch_size=32):
    """将tokenized的数据集分成指定大小的批次."""
    for i in range(0, len(tokenized_shard), batch_size):
        yield tokenized_shard[i:i + batch_size]

def process(example):
    ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out
# tokenize the dataset

def update_max_activations(activation_cache, activations, batch_ids, batch_index):
    """更新每个神经元的最大激活值."""
    activations_layer = activation_cache # ['features']  # 形状: (batch_size, seq_len, num_neurons)
    mean_activations = activations_layer.mean(dim=1)  # 形状: (batch_size, num_neurons)
    batch_size, seq_len, num_neurons = activations_layer.shape

    if MEAN_ACT == False:
        # 在 batch 维度上找到每个神经元的最大激活值 (形状: (num_neurons,))
        max_values_in_batch, max_positions = activations_layer.view(batch_size * seq_len, num_neurons).max(dim=0)

        # 将索引转换为 (batch, token) 索引
        example_indices, token_indices = np.divmod(max_positions.cpu().numpy(), seq_len)

        for neuron_idx in range(num_neurons):
            max_value = max_values_in_batch[neuron_idx]
            max_token_idx = batch_ids[example_indices[neuron_idx]][token_indices[neuron_idx]]
            if (max_value > activations[f"feature_{neuron_idx}"]["max_activation_value"]) and (NO_SAME_MAXTOKEN == False):
                activations[f"feature_{neuron_idx}"]["max_activation_value"] = max_value.item()
                activations[f"feature_{neuron_idx}"]["max_token_idx"] = batch_ids[example_indices[neuron_idx]][token_indices[neuron_idx]]
                activations[f"feature_{neuron_idx}"]["max_example_index"] = batch_index * batch_size + example_indices[neuron_idx]
                activations[f"feature_{neuron_idx}"]["max_token_index"] = token_indices[neuron_idx]
                activations[f"feature_{neuron_idx}"]["activations"] = activations_layer[example_indices[neuron_idx], :, neuron_idx].detach().clone()
            elif (max_value > activations[f"feature_{neuron_idx}"]["max_activation_value"]) and (NO_SAME_MAXTOKEN == True) and (max_token_idx not in max_tokens):
                activations[f"feature_{neuron_idx}"]["max_activation_value"] = max_value.item()
                activations[f"feature_{neuron_idx}"]["max_token_idx"] = batch_ids[example_indices[neuron_idx]][token_indices[neuron_idx]]
                activations[f"feature_{neuron_idx}"]["max_example_index"] = batch_index * batch_size + example_indices[neuron_idx]
                activations[f"feature_{neuron_idx}"]["max_token_index"] = token_indices[neuron_idx]
                activations[f"feature_{neuron_idx}"]["activations"] = activations_layer[example_indices[neuron_idx], :, neuron_idx].detach().clone()
                max_tokens[neuron_idx].append(max_token_idx)
    elif MEAN_ACT == True:
        max_mean_act_in_batch, max_positions = mean_activations.max(dim=0)
        for neuron_idx in range(num_neurons):
            if max_mean_act_in_batch[neuron_idx] > activations[f"feature_{neuron_idx}"]["mean_activation"]:
                activations[f"feature_{neuron_idx}"]["max_example_index"] = batch_index * batch_size + max_positions[neuron_idx]
                activations[f"feature_{neuron_idx}"]["activations"] = activations_layer[max_positions[neuron_idx], :, neuron_idx].detach().clone()
                activations[f"feature_{neuron_idx}"]["mean_activation"] = max_mean_act_in_batch[neuron_idx].item()


    # 将第一个example作为随机取样，记录activations
    if batch_index == 0:
        for neuron_idx in range(num_neurons):
            activations[f"feature_{neuron_idx}"]["activations_first_sample"] = activations_layer[0, :, neuron_idx].detach().clone()

def extract_selected_representation(encode_dict, selected_index):
    """
    根据 selected_index 和 encode_dict 提取形状为 (batch_size, seq_len, 768) 的张量。
    
    Args:
        encode_dict (dict): 包含 'inds' 和 'vals' 的字典，'inds' 和 'vals' 均为形状 (batch_size, seq_len, 32) 的张量。
        selected_index (list or torch.Tensor): 感兴趣的 768 个维度的索引。
    
    Returns:
        torch.Tensor: 形状为 (batch_size, seq_len, 768) 的张量。
    """
    inds = encode_dict['inds']  # 形状 (batch_size, seq_len, 32)
    vals = encode_dict['vals']  # 形状 (batch_size, seq_len, 32)

    # 将 selected_index 转换为 torch.Tensor（如果尚未转换）
    if not isinstance(selected_index, torch.Tensor):
        selected_index = torch.tensor(selected_index, dtype=torch.long)

    # 创建一个大小为 (32768,) 的张量，初始化为全零
    full_representation = torch.zeros(N_DIR, dtype=torch.float32, device=vals.device)

    # 构造稀疏表示向量
    batch_size, seq_len, k = inds.shape
    assert k==K
    output = torch.zeros(batch_size, seq_len, len(selected_index), device=vals.device)

    for b in range(batch_size):
        for s in range(seq_len):
            # 使用 'inds' 和 'vals' 重构 token 的原始表示向量
            full_representation.zero_()  # 重置为全零
            full_representation[inds[b, s]] = vals[b, s]

            # 提取感兴趣的 768 维度
            output[b, s] = full_representation[selected_index]

    return output

def process_shard(shard_index, dataset_sharded, start_event):
    start_event.wait()  # 等待主进程的信号
    file_name = save_file_name + f'__len_{max_length}__shard_{shard_index}.json'
    save_file = os.path.join(folder_path, file_name)
    if os.path.exists(save_file):
        print(f"File {save_file} already exists. Skipping...")
        return
    try:
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
            
            # comms = make_torch_comms(n_op_shards=cfg.n_op_shards, n_replicas=cfg.n_replicas)
            # ## dataloading is left as an exercise for the reader
            # acts_iter = load_activations(shard_num=8, batch_num=8, batch_step=8)    # 2^24 / (2^32 * 1.5)
            # stats_acts_sample = next(load_activations(shard_num=1, batch_num=1, batch_step=8))[:32768]
            
            # init_from_data_(autoencoder, stats_acts_sample, comms)
            autoencoder.to(device_gpt)

        # =======模型加载完成=======
        # -----------------------------------------------------------------------------
        tokenized_shard_part = dataset_sharded[f"train_shard_{shard_index}"].map(
            process,
            remove_columns=['text'],
            desc=f"tokenizing shard {shard_index}",
            num_proc=num_proc,
        )
        print(f"shard_{shard_index} is being processed")

        layers_to_hook = ['features']  # 捕捉的层

        # 用于存储每个神经元的激活值情况
        activations = {}
        for i in range(768):
            activations[f"feature_{i}"] = {
                "max_activation_value": torch.tensor(0.0).to(device_gpt),
                "max_token_idx": torch.tensor(0).to(device_gpt),
                "max_example_index": torch.tensor(0).to(device_gpt),
                "max_token_index": torch.tensor(0).to(device_gpt),
                "activations": torch.zeros(0).to(device_gpt),
                "mean_activation": torch.tensor(-100.0).to(device_gpt),
                "activations_first_sample": torch.zeros(0).to(device_gpt)
            }
        
        # 逐个batch处理
        for batch_idx, batch in enumerate(create_batches(tokenized_shard_part, batch_size=batch_size)):
        
            x = [torch.tensor(batch['ids'][i][:max_length]).to(device_gpt) for i in range(len(batch['ids']))]
            x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=50256).to(device_gpt)

            # 捕捉激活值
            logits, _, activation_cache = model.run_with_cache(x, layers_to_hook=layers_to_hook, device=device_gpt)
            input_tensor = activation_cache['features'].to('cuda')
            input_tensor_ln = input_tensor
            with torch.no_grad():
                encode_dict = autoencoder.encode(input_tensor_ln)
            activation_sae_selected = extract_selected_representation(encode_dict, selected_index)

            # 更新最大激活值
            update_max_activations(activation_sae_selected, activations, x, batch_idx)

            print(f"shard_{shard_index} batch_{batch_idx} finished")
            if batch_idx % 50 == 0:
                print(activations[f"feature_0"])
                print(activations[f"feature_1"])
            if batch_idx == MAX_BATCH_NUM:
                break
            # if shard_index <= 9 and batch_idx % 180 == 0:
            #     # 将 activations 转换为可序列化的字典
            #     json_ready_activations = convert_activations_to_json(activations)
            #     # 将其保存为 JSON
            #     with open(f'/data/jqliu/ML_jq/nanoGPT/activations/features_activations_shard_{shard_index}_batch_{batch_idx}.json', 'w') as json_file:
            #         json.dump(json_ready_activations, json_file, cls=NpEncoder)
            #     print(f"Shard {shard_index} batch {batch_idx} activations is saved")

        json_ready_activations = convert_activations_to_json(activations)
        # 将其保存为 JSON 文件
        
        with open(save_file, 'w') as json_file:
            json.dump(json_ready_activations, json_file, cls=NpEncoder)
        print(f"Shard {shard_index} batch {batch_idx} activations is saved")

    except Exception as e:
        # 捕获并输出错误信息
        print("发生异常:", e)
    finally:
        # 清理GPU资源和内存
        del model  # 删除模型以释放显存
        del activation_cache  # 删除激活缓存以释放显存
        torch.cuda.empty_cache()  # 清理未使用的显存
        gc.collect()  # 强制进行垃圾回收

# load the dataset
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

dataset_sharded = d.DatasetDict()

if __name__ == '__main__':
    # 创建事件对象
    manager = mp.Manager()
    start_event = manager.Event()
    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = d.load_dataset("/data2/datasets/openwebtext", num_proc=num_proc_load_dataset)
    """
    print(dataset)形如：
    
    DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 8013769
        })
    })
    """

    # Now we can shard the tokenized dataset into 64 parts
    num_shards = 64
    
    # Create 64 shards for each dataset split
    for split, dset in dataset.items():
        # 注意初始时split只有一个，即train
        sharded = dset.shard(num_shards=num_shards, index=0, contiguous=True).with_format('numpy')
        dataset_sharded[split] = sharded

        for shard_idx in range(num_shards):
            shard = dset.shard(num_shards=num_shards, index=shard_idx, contiguous=True)
            dataset_sharded[f"{split}_shard_{shard_idx}"] = shard


    mp.set_start_method('spawn', force=True)
     # 启动子进程
    with ProcessPoolExecutor(max_workers=2) as executor:
        try:
            futures = [executor.submit(process_shard, shard_idx, dataset_sharded, start_event) for shard_idx in range(num_shards)]
            # 初始化完成后设置事件
            start_event.set()

            # 等待所有子进程完成
            for future in futures:
                try:
                    future.result()  # 确保子进程完成，并抛出异常
                    gc.collect()  # 清理内存
                    torch.cuda.empty_cache()  # 清理GPU缓存
                except Exception as e:
                    print(f"Error in future: {e}")
        except Exception as e:
            print(f"错误：{e}")

        # Force garbage collection
        gc.collect()
        print("All shards processed")
