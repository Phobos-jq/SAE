"""
收集用于训练SAE的representation，存储成max_shard*12个json文件
最终一个json文件中含有8*batch_size*max_length个token的representation
"""
import os
import signal
from tqdm import tqdm
import numpy as np
import tiktoken
import datasets as d  # huggingface datasets
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
import gc
import json

max_length = 64
batch_size = 512
max_shard = 8
layers_to_hook = ['features']  # 如果要对中间层训SAE，改成layers_to_hood = ['transformer.h.{layer_num}.mlp.gelu']
save_file_path = '/data4/jqliu/ML_jq/SAE/activations/test'
model_dir = '/data4/jqliu/ML_jq/nanoGPT/out_ori/out_test'
enc = tiktoken.get_encoding("gpt2")

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

def process_shard(shard_index, dataset_sharded, start_event):
    start_event.wait()  # 等待主进程的信号
    try:
        init_from = 'resume' # either 'resume' (from an model_dir) or a gpt2 variant (e.g. 'gpt2-xl')
        temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
        seed = 1337
        device_gpt = 'cuda'
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
        compile = True # use PyTorch 2.0 to compile the model to be faster
        exec(open('configurator.py').read()) # overrides from command line or config file
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

        # =======模型加载完成=======
        # -----------------------------------------------------------------------------
        activations_forSAE = []

        tokenized_shard_part = dataset_sharded[f"train_shard_{shard_index}"].map(
            process,
            remove_columns=['text'],
            desc=f"tokenizing shard {shard_index}",
            num_proc=num_proc,
        )
        print(f"shard_{shard_index} is being processed")

        
        
        # 逐个batch处理
        for batch_idx, batch in enumerate(create_batches(tokenized_shard_part, batch_size=batch_size)):
        
            x = [torch.tensor(batch['ids'][i][:max_length]).to(device_gpt) for i in range(len(batch['ids']))]
            x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=50256).to(device_gpt)

            # 捕捉激活值
            logits, _, activation_cache = model.run_with_cache(x, layers_to_hook=layers_to_hook, device=device_gpt)

            activations = activation_cache[layers_to_hook[0]].reshape(-1, 768).tolist()
            activations_forSAE.append(activations)
            

            print(f"shard_{shard_index} batch_{batch_idx} finished")
            if batch_idx % 50 == 0:
                print(f"{len(activations)}, {len(activations[len(activations)-1])}")

            if (batch_idx+1)%8 == 0:
                 # 将其保存为 JSON 文件
                file_name = f'__len_{max_length}__batch_{batch_idx}__shard_{shard_index}.json'
                save_file = os.path.join(save_file_path, file_name)
                with open(save_file, 'w') as json_file:
                    json.dump(activations_forSAE, json_file, cls=NpEncoder)
                print(f"Shard {shard_index} batch {batch_idx} activations is saved")
                activations_forSAE = []
            if batch_idx == 95:
                break

        # # 将其保存为 JSON 文件
        # file_name = f'__len_{max_length}__shard_{shard_index}.json'
        # save_file = os.path.join(save_file_path, file_name)
        # with open(save_file, 'w') as json_file:
        #     json.dump(activations_forSAE, json_file, cls=NpEncoder)
        # print(f"Shard {shard_index} batch {batch_idx} activations is saved")

    except Exception as e:
        print(f"发生错误: {e}")

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
            futures = [executor.submit(process_shard, shard_idx, dataset_sharded, start_event) for shard_idx in range(max_shard)]
            # 初始化完成后设置事件
            start_event.set()

            # 等待所有子进程完成
            for future in futures:
                try:
                    future.result()  # 确保子进程完成，并抛出异常
                    gc.collect()  # 清理内存s
                    torch.cuda.empty_cache()  # 清理GPU缓存
                except Exception as e:
                    print(f"Error in future: {e}")
        except Exception as e:
            print(f"错误：{e}")

        # Force garbage collection
        gc.collect()
        print("All shards processed")
