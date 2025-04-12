import json
torch_2_jax_map = json.load(open('./load_params/map.json'))
from tqdm import tqdm
import numpy as np
import pathlib
from alphafold3.model import params
jax_params = params.get_model_haiku_params(model_dir=pathlib.Path('./models'))
print('type(jax_params)',type(jax_params))
import contextlib
import pathlib
import zstandard as zstd
import json  # 或者使用 pickle

import contextlib
import pathlib
import zstandard as zstd
import pickle  # 或者使用 pickle

@contextlib.contextmanager
def open_for_writing(output_file: pathlib.Path, is_compressed: bool):
    """
    用于写入文件的上下文管理器，支持压缩和非压缩模式。

    参数:
        output_file (pathlib.Path): 输出文件路径。
        is_compressed (bool): 是否启用压缩。
    """
    with open(output_file, 'wb') as f:
        if is_compressed:
            # 使用 zstandard 压缩写入
            compressor = zstd.ZstdCompressor()
            with compressor.stream_writer(f) as writer:
                yield writer
        else:
            # 直接写入文件
            yield f

def save_dict_to_file(data: dict, output_file: pathlib.Path, is_compressed: bool):
    """
    将字典保存到文件，支持压缩和非压缩模式。

    参数:
        data (dict): 要保存的字典。
        output_file (pathlib.Path): 输出文件路径。
        is_compressed (bool): 是否启用压缩。
    """
    # 将字典序列化为字节数据
    # serialized_data = json.dumps(data).encode('utf-8')  # 使用 json
    # 或者使用 pickle: 
    serialized_data = pickle.dumps(data)

    # 使用上下文管理器写入文件
    with open_for_writing(output_file, is_compressed) as f:
        f.write(serialized_data)


jax_name_list = [map_dict['jax_name'] for torch_name,map_dict in torch_2_jax_map.items()]
print("jax_params['__meta__']:",jax_params['__meta__'])
print("'__meta__' in jax_params",'__meta__' in jax_params)
jax_params_diffusion = {
    name: arr 
    for name,arr in jax_params.items() if name not in jax_name_list
}
jax_params_diffusion['__meta__'] = jax_params['__meta__']
save_path = 'torch_weigts/diffusion.bin'
with open(save_path,'wb') as f:
    pickle.dump(jax_params_diffusion, f)
# save_dict_to_file(jax_params_diffusion, save_path, True)

print('saved at', save_path)