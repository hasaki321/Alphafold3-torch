
import re

def size_to_array(size_str):
    size_str = ' '.join(size_str)
    # 使用正则表达式提取字符串中的数字
    matches = re.findall(r'\d+', size_str)
    # 将提取的字符串数字转换为整数，并返回列表
    return [int(num) for num in matches]


with open('load_params/jax_sorted_params.log','r') as f:
    jax_name_correct = f.read().strip().split('\n')
    jax_shape_list = [size_to_array(e.strip().split(' ')[2:]) for e in jax_name_correct if e.strip()]
    jax_name_correct = ['/'.join(e.strip().split(' ')[:2]) for e in jax_name_correct if e.strip()]

print('jax_name_correct',jax_name_correct)
print('jax_shape_list',jax_shape_list)
alignment = {
    # 第一级的四个模块
    'confidence_head':['diffuser/confidence_head'],
    'evoformer': ['diffuser/evoformer'],
    'evoformer_conditioning': ['diffuser/evoformer_conditioning_###'],
    'diffusion_head': ['diffuser/~/diffusion_head'],

    # confidence_head的第二级模块
    'distogram_feat_project':['~_embed_features/distogram_feat_project'],
    'left_target_feat_project':['~_embed_features/left_target_feat_project'],
    'right_target_feat_project':['~_embed_features/right_target_feat_project'],
    'experimentally_resolved_ln':['experimentally_resolved_ln'],
    'experimentally_resolved_logits':['experimentally_resolved_logits'],
    'left_distance_logits':['left_half_distance_logits'],
    'plddt_logits':['plddt_logits'],
    'plddt_logits_ln':['plddt_logits_ln'],
    'left_target_feat_project':['left_target_feat_project'],
    'left_target_feat_project':['left_target_feat_project'],
    'pairformer_stack':['__layer_stack_no_per_layer/confidence_pairformer'],
    'pae_logits':['pae_logits'],
    'pae_logits_ln':['pae_logits_ln'],
    'single_attention':['single_attention'],
    'logits_ln':['logits_ln'],
    'triangle_multiplication_incoming':['triangle_multiplication_incoming'],
    'center_norm':['center_norm'],
    'gate':['gate'],
    'gating_linear':['gating_linear'],
    'left_norm_input':['left_norm_input'],
    'input_layer_norm':['input_layer_norm'],
    'transition2':['transition2'],
    'single_pair_logits_projection':['single_pair_logits_projection'],

    # evoformer
    'bond_embedding':['bond_embedding'],
    'left_single':['left_single'],
    'msa_activations':['msa_activations'],
    'extra_msa_target_feat':['extra_msa_target_feat'],
    'msa_attention1':['msa_attention1'],
    'pair_norm':['pair_norm'],
    'msa_stack_layers':['__layer_stack_no_per_layer/msa_stack'],
    'pairformer_stack_layers':['__layer_stack_no_per_layer_1/trunk_pairformer'],
    'pair_attention1':['pair_attention1'],
    'pair_attention2':['pair_attention2'],
    'pair_transition':['pair_transition'],
    'template_embedding':['template_embedding'],
    'single_template_embedding':['single_template_embedding'],
    'triangle_multiplication_outgoing':['triangle_multiplication_outgoing'],
    'output_projection':['output_projection'],
    'projection':['projection'],
    'output_projection':['output_projection'],
    'act_norm':['act_norm'],
    'pair_logits':['pair_logits'],
    'pair_bias_projection':['pair_bias_projection'],
    'single_transition':['single_transition'],
    'template_embedding_iteration_layers':['__layer_stack_no_per_layer/template_embedding_iteration'],
    'position_activations':['~_relative_encoding/position_activations'],
    'prev_embedding_layer_norm':['prev_embedding_layer_norm'],
    'prev_single_embedding':['prev_single_embedding'],
    'right_single':['right_single'],
    'single_activations':['single_activations'],
    'query_embedding_norm':['query_embedding_norm'],
    'template_pair_embedding_layers':['template_pair_embedding_{}'],
    'output_layer_norm':['output_layer_norm'],
    'output_linear':['output_linear'],
    'msa_activations_proj':['msa_activations'],
    'outer_product_mean':['outer_product_mean'],
    'output_w':['output_w'],
    'output_b':['output_b'],
    'layer_norm_input':['layer_norm_input'],
    'left_projection':['left_projection'],
    'right_projection':['right_projection'],
    'msa_transition':['msa_transition'],
    'prev_embedding_proj':['prev_embedding'],
    'single_activations_proj':['single_activations'],
    'prev_single_embedding_layer_norm':['prev_single_embedding_layer_norm'],
    'prev_single_embedding_proj':['prev_single_embedding'],
    'single_pair_logits_norm':['single_pair_logits_norm'],
    'prev_embedding':['prev_embedding'],

    # evoformer_conditioning
    'cross_att_blocks':['__layer_stack_with_per_layer'],
    'transition1':['transition1'],
    'pair_input_layer_norm':['pair_input_layer_norm'],

    # diffusion head
    'super_transformer_blocks':['__layer_stack_with_per_layer/__layer_stack_with_per_layer'],
    'transformer_blocks':['__layer_stack_with_per_layer'],
    
    'embed_pair_offsets_valid':['embed_pair_offsets_valid'],
    'project_atom_features_for_aggr':['project_atom_features_for_aggr'],
    'q_projection_bias':['q_projection/bias'],
    # other
    '_per_atom_conditioning': [''],  # Implicit in JAX

    'atom_cross_att_encoder': ['diffuser/~/diffusion_head','diffusion_atom_encoder','evoformer_conditioning_atom_encoder'],
    'atom_features_layer_norm': ['diffusion_atom_features_layer_norm'],
    'atom_features_to_position_update': ['diffusion_atom_features_to_position_update'],
    'atom_transformer_decoder': ['diffusion_atom_transformer_decoder'],
    'atom_transformer_encoder': ['atom_transformer_encoder'],
    'bias': ['bias'],  # Implicit in JAX
    'blocks': ['__layer_stack_with_per_layer','__layer_stack_no_per_layer', ],
    'super_blocks': ['__layer_stack_with_per_layer/__layer_stack_with_per_layer'],
    'pair_attention1': ['pair_attention1'],
    'pair_attention2': ['pair_attention2'],
    'diffusion_atom_cross_att_decoder': ['diffusion_atom_transformer_decoder'],
    'diffusion_atom_cross_att_encoder': ['diffusion_###', 'evoformer_conditioning_atom_transformer_encoder'],
    'embed_pair_distances': ['embed_pair_distances', 'diffusion_embed_pair_distances_1', 'embed_pair_distances_1'],
    'embed_pair_offsets': ['embed_pair_offsets', 'embed_pair_offsets_1', 'diffusion_embed_pair_offsets_valid', 'embed_pair_offsets_valid'],
    'embed_ref_atom_name': ['embed_ref_atom_name'],
    'embed_ref_charge': ['embed_ref_charge'],
    'embed_ref_element': ['embed_ref_element'],
    'embed_ref_mask': ['embed_ref_mask'],
    'embed_ref_pos': ['embed_ref_pos'],
    # 'embed_trunk_pair_cond': ['diffusion_embed_trunk_pair_cond'],
    # 'atom_positions_to_features': ['diffusion_atom_positions_to_features'],
    # 'embed_trunk_single_cond': ['diffusion_embed_trunk_single_cond'],
    # 'lnorm_trunk_pair_cond': ['diffusion_lnorm_trunk_pair_cond'],
    # 'lnorm_trunk_single_cond': ['diffusion_lnorm_trunk_single_cond'],
    'lnorm_trunk_single_cond': ['lnorm_trunk_single_cond'],
    'lnorm_trunk_pair_cond': ['lnorm_trunk_pair_cond'],
    'atom_positions_to_features': ['atom_positions_to_features'],
    'embed_trunk_pair_cond': ['embed_trunk_pair_cond'],
    'embed_trunk_single_cond': ['embed_trunk_single_cond'],

    'ffw_transition1': ['ffw_transition1'],
    'single_transition_blocks': ['single_transition_{}ffw_###'],
    'gating_query': ['gating_query', 'gating_linear', 'single_attention_gating_query'],
    'k_projection': ['k_projection', 'single_attention_k_projection'],
    'noise_embedding_initial_norm': ['noise_embedding_initial_norm'],
    'noise_embedding_initial_projection': ['noise_embedding_initial_projection'],
    'output_norm': ['output_norm', 'output_layer_norm'],
    'pair_cond_initial_norm': ['pair_cond_initial_norm'],
    'pair_cond_initial_projection': ['pair_cond_initial_projection'],
    'pair_logits_projection': ['pair_logits_projection/weights'],
    'pair_transition_blocks': ['pair_transition_{}ffw_###', 'pair_transition_0ffw_layer_norm', 'pair_transition_0ffw_transition1', 'pair_transition_0ffw_transition2', 'pair_transition_1ffw_layer_norm', 'pair_transition_1ffw_transition1', 'pair_transition_1ffw_transition2'],
    'project_token_features_for_broadcast': ['diffusion_project_token_features_for_broadcast', 'diffusion_project_token_features_for_broadcast'],
    'self_attention': ['transformer###', 'single_attention_layer_norm', 'single_attention_transition2', 'single_attention_v_projection'],
    'single_cond_bias': ['single_cond_bias', 'ffw_single_cond_bias', 'ksingle_cond_bias', 'qsingle_cond_bias'],
    'single_cond_embedding_norm': ['single_cond_embedding_norm'],
    'single_cond_embedding_projection': ['single_cond_embedding_projection'],
    'single_cond_initial_norm': ['single_cond_initial_norm'],
    'single_cond_initial_projection': ['single_cond_initial_projection'],
    'single_cond_layer_norm': ['single_cond_layer_norm', 'ffw_single_cond_layer_norm', 'ksingle_cond_layer_norm', 'qsingle_cond_layer_norm'],
    'single_cond_scale': ['single_cond_scale', 'ffw_single_cond_scale', 'ksingle_cond_scale', 'qsingle_cond_scale'],
    'transformer': ['transformer'],
    # 'transition': ['diffusion_atom_transformer_encoder### or diffusion_atom_transformer_decoder###'],
    'v_projection': ['v_projection/weights', 'single_attention_v_projection'],
    'weight': ['weights'], # Implicit in JAX
    'k_projection':['k_projection/weights'],
    'q_projection': ['q_projection/weights', 'single_attention_q_projection'],
    # transition_block
    'adaptive_zero_init':['###'],
    'adaptive_zero_cond':['adaptive_zero_cond'],
    'project_atom_features_for_aggr': ['project_atom_features_for_aggr'],
    'layer_norm':['layer_norm'],


    # 需要根据前面的内容确定替换的内容，在post_process中被处理
    'cross_attention': ['cross_attention'], # diffusion_atom_transformer_encoder### or diffusion_atom_transformer_decoder###
    'transition': ['transition'], # ['diffusion_atom_transformer_encoder### or diffusion_atom_transformer_decoder###'],
    'transition_block':['transition_block'],
    # 'transition_block':['__layer_stack_with_per_layer/transformer### or __layer_stack_with_per_layer/diffusion_atom_transformer_encoder###'],
    'pair_mlp': ['pair_mlp_{}'], # diffusion_pair_mlp_{} or evoformer_conditioning_pair_mlp_{}
    'single_to_pair_cond_col': ['single_to_pair_cond_col', 'single_to_pair_cond_col_1', 'diffusion_single_to_pair_cond_col', 'diffusion_single_to_pair_cond_col_1'],
    'single_to_pair_cond_row': ['single_to_pair_cond_row', 'single_to_pair_cond_row_1', 'diffusion_single_to_pair_cond_row', 'diffusion_single_to_pair_cond_row_1'],
    'adaptive_layernorm': ['###', 'ffw_adaptive_zero_cond'],
    # 'adaptive_layernorm': [''],
    'adaptive_layernorm_k': ['k###', 'ffw_adaptive_zero_cond'],
    'adaptive_layernorm_q': ['q###', 'ffw_adaptive_zero_cond'],
    # 数字的处理
    **{
        str(num):num for num in range(50)
    }
}

def compare_same_shape(torch_shape, jax_shape):
    jax_sub_shape = jax_shape[-len(torch_shape):]
    flag = None
    if len(torch_shape) == 1:
        flag = torch_shape==jax_sub_shape
    elif len(torch_shape) == 2:
        flag = torch_shape==jax_sub_shape[::-1]
    elif len(torch_shape) == 3:
        flag = [torch_shape[1],torch_shape[2],torch_shape[0]]==jax_sub_shape or torch_shape == jax_sub_shape
    return flag

def post_process(jax_name:str, block_index_list:list):
    # replace_list = ['k','###']
    # for e in replace_list:
    #     if f'{e}/' in jax_name:
    #         jax_name = jax_name.replace(f'{e}/',)

    # if '/k/' in jax_name:
    #     jax_name = jax_name.replace('/k/','/k')
    if '###/' in jax_name:
        jax_name = jax_name.replace('###/','')
    if 'cross_attention' in jax_name:
        pos = jax_name.find('cross_attention')
        if 'diffusion_atom_transformer_encoder' in jax_name[:pos]:
            jax_name = jax_name.replace('cross_attention','diffusion_atom_transformer_encoder###')
        elif 'diffusion_atom_transformer_decoder' in jax_name[:pos]:
            jax_name = jax_name.replace('cross_attention','diffusion_atom_transformer_decoder###')
        elif 'evoformer_conditioning_atom_transformer_encoder' in jax_name[:pos]:
            jax_name = jax_name.replace('cross_attention','evoformer_conditioning_atom_transformer_encoder###')
        else:
            raise ValueError(f'not match prefix found in {jax_name}. It should have diffusion_atom_transformer_encoder### or diffusion_atom_transformer_decoder### or evoformer_conditioning_atom_transformer_encoder###')
    if '/transition/' in jax_name:
        pos = jax_name.find('transition')
        if 'diffusion_atom_transformer_encoder' in jax_name[:pos]:
            jax_name = jax_name.replace('transition','diffusion_atom_transformer_encoder###')
        elif 'diffusion_atom_transformer_decoder' in jax_name[:pos]:
            jax_name = jax_name.replace('transition','diffusion_atom_transformer_decoderffw_###')
        else:
            raise ValueError(f'not match prefix found in {jax_name}. It should have diffusion_atom_transformer_encoder### or diffusion_atom_transformer_decoder### ')
    if '/transition_block/' in jax_name:
        pos = jax_name.find('/transition_block/')
        if 'diffusion_atom_transformer_encoder' in jax_name[:pos]:
            jax_name = jax_name.replace('transition_block','__layer_stack_with_per_layer/diffusion_atom_transformer_encoderffw_###')
        elif 'diffuser/evoformer_conditioning_atom_transformer_encoder' in jax_name[:pos]:
            jax_name = jax_name.replace('transition_block','evoformer_conditioning_atom_transformer_encoderffw_###')
        elif 'diffuser/~/diffusion_head/transformer' in jax_name[:pos]:
            jax_name = jax_name.replace('transition_block','transformer###')
        elif 'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder' in jax_name:
            jax_name = jax_name.replace('transition_block','diffusion_atom_transformer_decoderffw_###')
        else:
            raise ValueError(f'not match prefix found in {jax_name}. It should have diffusion_atom_transformer_encoder### or transformer### or evoformer_conditioning_atom_transformer_encoder###')
    
    if '/single_attention/' in jax_name:
        if 'diffuser/evoformer/__layer_stack_no_per_layer' in jax_name:
            jax_name = jax_name.replace('/single_attention/','/single_attention_')
        else:
            raise ValueError(f'not match prefix found in {jax_name}. It should have diffuser/evoformer/__layer_stack_no_per_layer')
    
    if '/adaptive_layernorm/' in jax_name:
        if 'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder' in jax_name:
            jax_name = jax_name.replace('/adaptive_layernorm/','/diffusion_atom_transformer_decoder')
        else:
            raise ValueError(f'not match prefix found in {jax_name}. It should have diffusion_atom_transformer_decoder')
    if '/project_atom_features_for_aggr/' in jax_name:
        if 'diffuser/~/diffusion_head/' in jax_name:
            jax_name = jax_name.replace("project_atom_features_for_aggr",'diffusion_project_atom_features_for_aggr')
        elif 'diffuser/evoformer_conditioning_' in jax_name:
            pass
        else:
            raise ValueError


    # if 'single_to_pair_cond_col' in jax_name:
    #     if jax_name.startswith('diffuser/~/diffusion_head/'):
    #         jax_name = jax_name.replace('single_to_pair_cond_col','diffusion_single_to_pair_cond_col{}')
    #     elif jax_name.startswith('diffuser/'):
    #         jax_name = jax_name.replace('single_to_pair_cond_col','evoformer_conditioning_single_to_pair_cond_col{}')
    #     else:
    #         raise ValueError(f'not match prefix found in {jax_name}. It should have diffuser/~/diffusion_head/ or diffuser/')
    
    # if 'single_to_pair_cond_row' in jax_name:
    #     if jax_name.startswith('diffuser/~/diffusion_head/'):
    #         jax_name = jax_name.replace('single_to_pair_cond_row','diffusion_single_to_pair_cond_row_{}')
    #     elif jax_name.startswith('diffuser/'):
    #         jax_name = jax_name.replace('single_to_pair_cond_row','evoformer_conditioning_single_to_pair_cond_row_{}')
    #     else:
    #         raise ValueError(f'not match prefix found in {jax_name}. It should have diffuser/~/diffusion_head/ or diffuser/')
    
    if '//' in jax_name:
        jax_name = jax_name.replace('//','/')

    if '###/' in jax_name:
        jax_name = jax_name.replace('###/','')



    replace_dict = {
        ### layer norm 的名字换
        'norm/weights':'norm/scale',
        'norm/bias':'norm/offset',
        'left_norm_input/weights':'left_norm_input/scale',
        'left_norm_input/bias':'left_norm_input/offset',
        'ln/bias':'ln/offset',
        'ln/weights':'ln/scale',
        'layer_norm_input/bias':'layer_norm_input/offset',
        'layer_norm_input/weights':'layer_norm_input/scale',
        'lnorm_trunk_pair_cond/bias':'lnorm_trunk_pair_cond/offset',
        'lnorm_trunk_pair_cond/weights':'lnorm_trunk_pair_cond/scale',
        'lnorm_trunk_single_cond/weights':'lnorm_trunk_single_cond/scale',

        'single_attention_ffw_':'single_attention_',
        'single_attention_single_cond_layer_norm':'single_attention_layer_norm',
        'diffusion_atom_transformer_decoder/diffusion_atom_transformer_decoder':'diffusion_atom_transformer_decoder',
        'evoformer_conditioning_/':'evoformer_conditioning_',
        'diffuser/confidence_head/left_target_feat_project':'diffuser/confidence_head/~_embed_features/left_target_feat_project',
        'diffuser/confidence_head/right_target_feat_project':'diffuser/confidence_head/~_embed_features/right_target_feat_project',
        'diffusion_/':'diffusion_',
        'diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/transformer':'diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoder',
        'diffusion_atom_transformer_encoder/__layer_stack_with_per_layer/__layer_stack_with_per_layer/':'diffusion_atom_transformer_encoder/__layer_stack_with_per_layer/',
        'diffusion_atom_transformer_decoder/diffusion_atom_features_layer_norm':'diffusion_atom_features_layer_norm',
        'diffusion_atom_transformer_decoder/diffusion_atom_features_layer_norm':'diffusion_atom_features_layer_norm',
        'diffusion_atom_transformer_decoder/diffusion_atom_features_to_position_update':'diffusion_atom_features_to_position_update',
        'diffusion_atom_transformer_decoder/diffusion_project_token_features_for_broadcast':'diffusion_project_token_features_for_broadcast',
        'pair_transition_{}ffw_single_cond_layer_norm':'pair_transition_0ffw_layer_norm',
        'single_transition_{}ffw_single_cond_layer_norm':'single_transition_0ffw_layer_norm',
        'pair_transition_{}transition2':'pair_transition_{}ffw_transition2',
        'single_transition_{}transition2':'single_transition_{}ffw_transition2',
        'ffw_ffw_':'ffw_',
    }
    for k in replace_dict:
        if k in jax_name:
            jax_name = jax_name.replace(k,replace_dict[k])
    
    if 'diffusion_atom_transformer_encoder/__layer_stack_with_per_layer/msa_transition' in jax_name:
        jax_name = jax_name.replace('diffusion_atom_transformer_encoder/__layer_stack_with_per_layer/msa_transition/','diffusion_atom_transformer_encoder/__layer_stack_with_per_layer/diffusion_atom_transformer_encoder')
    if '{}' in jax_name and len(block_index_list):
        jax_name = jax_name.replace('{}',f'{block_index_list.pop(0)}')
        if 'pair_mlp' in jax_name:
            jax_name = jax_name.replace('pair_mlp_3','pair_mlp_2')
            jax_name = jax_name.replace('pair_mlp_5','pair_mlp_3')

    return jax_name

def process_torch_name(torch_name):
    elements = torch_name.strip().split(' ')[0].split('.')
    
    mapped_elements = []
    block_index_list = []
    for e in elements:
        if e not in alignment:
            raise ValueError(f'not found {e} in alignment map')
        elif isinstance(alignment[e],int):
            block_index_list.append(alignment[e])
        else:
            mapped_elements.append(alignment[e][0])
    mapped_jax_name = '/'.join(mapped_elements)
    return mapped_jax_name, block_index_list

torch_jax_name_list = []
block_index_list_list = []
torch_shape_list = []
with open('load_params/torch_state_dict.log','r') as f:
    for line in f.readlines():
        if line.strip():
            torch_name = line.strip()
            mapped_jax_name, block_index_list = process_torch_name(torch_name)
            jax_name = post_process(mapped_jax_name, block_index_list)
            torch_shape_list.append(size_to_array(torch_name.split(' ')[1:]))
            if torch_name.split(' ')[0] == 'evoformer_conditioning.embed_pair_distances.weight':
                jax_name = 'diffuser/evoformer_conditioning_embed_pair_distances_1/weights'
            if torch_name.split(' ')[0] == 'evoformer_conditioning.embed_pair_offsets.weight':
                jax_name = 'diffuser/evoformer_conditioning_embed_pair_offsets_1/weights'
            if torch_name.split(' ')[0] == 'evoformer_conditioning.single_to_pair_cond_col.weight':
                jax_name = 'diffuser/evoformer_conditioning_single_to_pair_cond_col_1/weights'
            if torch_name.split(' ')[0] == 'evoformer_conditioning.single_to_pair_cond_row.weight':
                jax_name = 'diffuser/evoformer_conditioning_single_to_pair_cond_row_1/weights'
            
            torch_jax_name_list.append((torch_name,jax_name))
            block_index_list_list.append(block_index_list)


# 使用 zip 将两个列表组合在一起
combined = list(zip(torch_jax_name_list, block_index_list_list, torch_shape_list))

# 根据 torch_jax_name_list 中的 jax_name 排序
combined_sorted = sorted(combined, key=lambda x: x[0][1])


jax_name_correct_checked = {e:False for e in jax_name_correct}
jax_name_correct_shape_map = {name:shape for name,shape in zip(jax_name_correct,jax_shape_list)}
torch_2_jax_map = {

}
for (torch_name, jax_name), block_index_list, torch_shape in combined_sorted:
    print(torch_name)
    print(jax_name)
    if jax_name in jax_name_correct:
        print('aligned')
        jax_name_correct_checked[jax_name] = True
    else:
        print('not aligned')
    if compare_same_shape(torch_shape, jax_name_correct_shape_map[jax_name]):
        print('shape aligned', torch_shape)
    else:
        print('shape not aligned, torch_shape', torch_shape, 'jax_shape', jax_name_correct_shape_map[jax_name])
    torch_2_jax_map[torch_name.split(' ')[0]] = {
        'jax_name': jax_name,
        'block_index_list': block_index_list,
        'torch_shape':torch_shape,
    }
    print()


print('no appearance in torch')
for k,v in jax_name_correct_checked.items():
    if not v:
        print(k)

import json

with open('load_params/map.json','w') as f:
    json.dump(torch_2_jax_map,f,indent=4)
