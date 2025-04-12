import torch.utils.dlpack
from alphafold3.model import params
import json
import torch

torch_2_jax_map = json.load(open('load_params/map.json','r'))
# {
#     "evoformer.msa_stack_layers.0.msa_attention1.act_norm.bias": {
#         "jax_name": "diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/msa_attention1/act_norm/offset",
#         "block_index_list": [
#             0
#         ],
#        "torch_shape": [
#             64
#        ],
#     },
#     ...
# }
from tqdm import tqdm
import numpy as np
import pathlib
import jax
jax_params = params.get_model_haiku_params(model_dir=pathlib.Path('./models'))
torch_state_dict = {

}


for torch_name, map_dict in tqdm(torch_2_jax_map.items(),total = len(torch_2_jax_map)):
    jax_name = map_dict['jax_name']
    block_index_list = map_dict['block_index_list']
    torch_shape = map_dict['torch_shape']
    jax_arr = jax_params['/'.join(jax_name.split('/')[:-1])][jax_name.split('/')[-1]]

    # 步骤 1：将 JAX 数组转换为 DLPack 胶囊对象
    dlpack_capsule = jax.dlpack.to_dlpack(jax_arr)

    # 步骤 2：从 DLPack 胶囊对象创建 PyTorch Tensor，保留原始类型

    torch_tensor = torch.from_dlpack(dlpack_capsule)
    # jax_arr = np.array(jax_arr).astype(np.float32)
    print(jax_name,jax_arr.dtype,jax_arr.shape, torch_tensor.dtype)
    jax_shape = list(jax_arr.shape)

    # torch_tensor = None
    if jax_name in  [f'diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_{i}/weights' for i in [1,4,5,6,7]]:
        torch_tensor = torch_tensor.unsqueeze(1)
    elif jax_name == 'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_attention_q_projection/bias':
        assert len(block_index_list)==1
        torch_tensor = torch_tensor[block_index_list[0]].unsqueeze(0)
    elif jax_name == 'diffuser/evoformer_conditioning_atom_transformer_encoder/__layer_stack_with_per_layer/evoformer_conditioning_atom_transformer_encoderq_projection/bias':
        assert len(block_index_list)==1
        torch_tensor = torch_tensor[block_index_list[0]].unsqueeze(0)
    else:
        assert len(torch_shape) + len(block_index_list) == len(jax_shape), f'not match between {torch_shape} and {block_index_list} and {jax_shape} in {jax_name} '
        jax_sub_shape = jax_shape
        torch_sub_arr = torch_tensor
        if len(block_index_list) > 0:
            jax_sub_shape = jax_shape[-len(torch_shape):]
        if len(block_index_list) == 1:
            torch_sub_arr = torch_sub_arr[block_index_list[0]]
        if len(block_index_list) == 2:
            torch_sub_arr = torch_sub_arr[block_index_list[0]][block_index_list[1]]
        if len(torch_shape) == 1:
            torch_tensor = torch_sub_arr
        elif len(torch_shape) == 2:
            torch_tensor = torch_sub_arr.transpose(0,1)
        elif len(torch_shape) == 3:
            if torch_shape == jax_sub_shape:
                torch_tensor = torch_sub_arr
            elif [torch_shape[1],torch_shape[2],torch_shape[0]]==jax_sub_shape:
                torch_tensor = torch_sub_arr.permute(2, 0, 1)
            else:
                raise ValueError
    assert list(torch_tensor.shape) == torch_shape, f'list(torch_tensor.shape) is {list(torch_tensor.shape)} torch_shape is {torch_shape},  jax_sub_shape is {jax_sub_shape} in {jax_name}'
    torch_state_dict[torch_name] = torch_tensor
save_path = 'torch_weigts/evo.pth'
torch.save(torch_state_dict,save_path)
print('saved at',save_path)
