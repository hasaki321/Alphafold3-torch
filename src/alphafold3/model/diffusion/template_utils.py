from alphafold3.model import protein_data_processing
import jax.numpy as jnp
import jax
import numpy as np
import torch
from alphafold3.common import base_config
from alphafold3.jax import geometry
from typing import Optional, Tuple, Union, Callable, List, Self
from jaxlib.xla_extension import ArrayImpl


Array = jnp.ndarray | np.ndarray

jax2th = lambda x: torch.from_dlpack(jax.dlpack.to_dlpack(x)) if isinstance(x, (jnp.ndarray, ArrayImpl)) else x
th2jax = lambda x: jax.dlpack.from_dlpack(torch.to_dlpack(x)) if isinstance(x, torch.Tensor) else x
th2jax_np = lambda x: jnp.array(x.numpy()) if isinstance(x, torch.Tensor) else x
align_error = lambda x, y, name='': print(f'{name} align error:', 
                                          jnp.mean(jnp.abs(th2jax(x)-th2jax(y)))/jnp.mean(jnp.abs(th2jax(x)))*100, 
                                          '%')

def torch_wrapper(fn: Callable):
    def wrapped(*args, **kwargs):
        # 转换位置参数中的JAX数组为PyTorch张量
        converted_args = [
            jax2th(arg) if isinstance(arg, (jnp.ndarray, ArrayImpl)) else arg
            for arg in args
        ]
        converted_args = [
            torch.from_numpy(arg) if isinstance(arg, np.ndarray) else arg
            for arg in args
        ]
        # 转换关键字参数中的JAX数组为PyTorch张量
        converted_kwargs = {
            key: jax2th(value) if isinstance(value, (jnp.ndarray, ArrayImpl)) else value
            for key, value in kwargs.items()
        }
        converted_kwargs = {
            key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in kwargs.items()
        }
        
        # 调用原始函数并返回结果
        return fn(*converted_args, **converted_kwargs)
    
    return wrapped

def jax_wrapper(fn: Callable):
    def wrapped(*args, **kwargs):
        converted_args = [
            th2jax(arg) if isinstance(arg, torch.Tensor) else arg
            for arg in args
        ]
        
        converted_kwargs = {
            key: th2jax(value) if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        
        # 调用原始函数并返回结果
        return fn(*converted_args, **converted_kwargs)
    
    return wrapped
  


def pseudo_beta_fn(
    aatype: Array,
    dense_atom_positions: Array,
    dense_atom_masks: Array,
    is_ligand: Array | None = None,
    use_jax: bool | None = True,
) -> tuple[Array, Array] | Array:
  """Create pseudo beta atom positions and optionally mask.

  Args:
    aatype: [num_res] amino acid types.
    dense_atom_positions: [num_res, NUM_DENSE, 3] vector of all atom positions.
    dense_atom_masks: [num_res, NUM_DENSE] mask.
    is_ligand: [num_res] flag if something is a ligand.
    use_jax: whether to use jax for the computations.

  Returns:
    Pseudo beta dense atom positions and the corresponding mask.
  """
  if use_jax:
    xnp = jnp
  else:
    xnp = np

  if is_ligand is None:
    is_ligand = xnp.zeros_like(aatype)

  pseudobeta_index_polymer = xnp.take(
      protein_data_processing.RESTYPE_PSEUDOBETA_INDEX, aatype, axis=0
  ).astype(xnp.int32)

  pseudobeta_index = jnp.where(
      is_ligand,
      jnp.zeros_like(pseudobeta_index_polymer),
      pseudobeta_index_polymer,
  )

  pseudo_beta = xnp.take_along_axis(
      dense_atom_positions, pseudobeta_index[..., None, None], axis=-2
  )
  pseudo_beta = xnp.squeeze(pseudo_beta, axis=-2)

  pseudo_beta_mask = xnp.take_along_axis(
      dense_atom_masks, pseudobeta_index[..., None], axis=-1
  ).astype(xnp.float32)
  pseudo_beta_mask = xnp.squeeze(pseudo_beta_mask, axis=-1)

  return pseudo_beta, pseudo_beta_mask

def pseudo_beta_fn_th(
    aatype: torch.Tensor,
    dense_atom_positions: torch.Tensor,
    dense_atom_masks: torch.Tensor,
    is_ligand: torch.Tensor | None = None,
    device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch 实现的伪 beta 原子位置和掩码生成函数

    Args:
        aatype: [num_res] 氨基酸类型
        dense_atom_positions: [num_res, NUM_DENSE, 3] 原子位置
        dense_atom_masks: [num_res, NUM_DENSE] 原子掩码
        is_ligand: [num_res] 配体标记
        device: 计算设备

    Returns:
        pseudo_beta: [num_res, 3] 伪 beta 原子位置
        pseudo_beta_mask: [num_res] 伪 beta 掩码
    """
    # 确保常量数组在指定设备上
    restype_pseudobeta_index = torch.tensor(
        protein_data_processing.RESTYPE_PSEUDOBETA_INDEX,
        dtype=torch.long,
        device=device
    )

    # 处理 is_ligand 缺省值
    if is_ligand is None:
        is_ligand = torch.zeros_like(aatype, device=device)

    # 获取基础伪 beta 索引
    pseudobeta_index_polymer = restype_pseudobeta_index[aatype.long()]

    # 应用配体掩码
    pseudobeta_index = torch.where(
        is_ligand.bool(),
        torch.zeros_like(pseudobeta_index_polymer),
        pseudobeta_index_polymer
    )

    # 调整索引形状用于 gather 操作 [num_res, 1, 1]
    gather_idx = pseudobeta_index[..., None, None].long()

    # 收集伪 beta 原子位置 [num_res, 1, 3] -> [num_res, 3]
    pseudo_beta = torch.gather(
        dense_atom_positions,
        dim=-2,
        index=gather_idx.expand(-1, -1, 3)
    ).squeeze(-2)

    # 收集伪 beta 原子掩码 [num_res, 1] -> [num_res]
    pseudo_beta_mask = torch.gather(
        dense_atom_masks,
        dim=-1,
        index=gather_idx.squeeze(-1)
    ).to(torch.float32).squeeze(-1)

    return pseudo_beta, pseudo_beta_mask


class DistogramFeaturesConfig(base_config.BaseConfig):
  # The left edge of the first bin.
  min_bin: float = 3.25
  # The left edge of the final bin. The final bin catches everything larger than
  # `max_bin`.
  max_bin: float = 50.75
  # The number of bins in the distogram.
  num_bins: int = 39


def dgram_from_positions(positions, config: DistogramFeaturesConfig):
  """Compute distogram from amino acid positions.

  Args:
    positions: (num_res, 3) Position coordinates.
    config: Distogram bin configuration.

  Returns:
    Distogram with the specified number of bins.
  """
  lower_breaks = jnp.linspace(config.min_bin, config.max_bin, config.num_bins)
  lower_breaks = jnp.square(lower_breaks)
  upper_breaks = jnp.concatenate(
      [lower_breaks[1:], jnp.array([1e8], dtype=jnp.float32)], axis=-1
  )
  dist2 = jnp.sum(
      jnp.square(
          jnp.expand_dims(positions, axis=-2)
          - jnp.expand_dims(positions, axis=-3)
      ),
      axis=-1,
      keepdims=True,
  )

  dgram = (dist2 > lower_breaks).astype(jnp.float32) * (
      dist2 < upper_breaks
  ).astype(jnp.float32)
  return dgram

def dgram_from_positions_th(positions: torch.Tensor, config: DistogramFeaturesConfig) -> torch.Tensor:
  """Compute distogram from amino acid positions.

  Args:
    positions: (num_res, 3) Position coordinates.
    config: Distogram bin configuration.

  Returns:
    Distogram with the specified number of bins.
  """
  lower_breaks = torch.linspace(config.min_bin, config.max_bin, config.num_bins) # 使用 torch.linspace
  lower_breaks = torch.square(lower_breaks) # 使用 torch.square
  upper_breaks = torch.cat(
      [lower_breaks[1:], torch.tensor([1e8], dtype=torch.float32)], dim=-1 # 使用 torch.cat 和 torch.tensor
  )
  dist2 = torch.sum(
      torch.square(
          positions[:, None, :] - positions[None, :, :] # 使用 unsqueeze 和 torch.square
      ),
      dim=-1,
      keepdim=True,
  )

  dgram = (dist2 > lower_breaks).float() * ( # 使用 .float() 转换类型
      dist2 < upper_breaks
  ).float()
  return dgram

def make_backbone_rigid(
    positions: geometry.Vec3Array,
    mask: jnp.ndarray,
    group_indices: jnp.ndarray,
) -> tuple[geometry.Rigid3Array, jnp.ndarray]:
  """Make backbone Rigid3Array and mask.

  Args:
    positions: (num_res, num_atoms) of atom positions as Vec3Array.
    mask: (num_res, num_atoms) for atom mask.
    group_indices: (num_res, num_group, 3) for atom indices forming groups.

  Returns:
    tuple of backbone Rigid3Array and mask (num_res,).
  """
  backbone_indices = group_indices[:, 0]

  # main backbone frames differ in sidechain frame convention.
  # for sidechain it's (C, CA, N), for backbone it's (N, CA, C)
  # Hence using c, b, a, each of shape (num_res,).
  c, b, a = [backbone_indices[..., i] for i in range(3)]

  slice_index = jax.vmap(lambda x, i: x[i])
  # print('mask',slice_index(mask, a).shape)
  rigid_mask = (
      slice_index(mask, a) * slice_index(mask, b) * slice_index(mask, c)
  ).astype(jnp.float32)
  # print(rigid_mask.shape)
  # print(positions.x.shape)

  frame_positions = []
  for indices in [a, b, c]:
    print('pos',slice_index(positions.x, indices).shape)
    frame_positions.append(
        jax.tree.map(lambda x, idx=indices: slice_index(x, idx), positions)
    )
  # print(frame_positions[0].x.shape)

  rotation = geometry.Rot3Array.from_two_vectors(
      frame_positions[2] - frame_positions[1],
      frame_positions[0] - frame_positions[1],
  )
  # print(rotation.xx.shape)
  rigid = geometry.Rigid3Array(rotation, frame_positions[1])

  return rigid, rigid_mask

@torch.compiler.disable()
def make_backbone_rigid_th(
    dense_atom_positions: torch.Tensor,  # 假设 geometry.Vec3Array 已转换为 PyTorch
    mask: torch.Tensor,
    group_indices: torch.Tensor,
) -> Tuple[geometry.Rigid3Array, torch.Tensor]:  # 假设 geometry.Rigid3Array 已转换为 PyTorch
  dense_atom_positions = th2jax_np(dense_atom_positions)
#   print(dense_atom_positions)
#   print(type(dense_atom_positions))
  positions = geometry.Vec3Array.from_array(dense_atom_positions)
  rigid, rigid_mask = make_backbone_rigid(
    positions,
    th2jax_np(mask),
    th2jax_np(group_indices),
  )

  points = rigid.translation
  rigid_vec = rigid[:, None]
  # print(rigid_vec.rotation.xx.shape)
  # print(rigid_vec.translation.x.shape)
  rigid_vec = rigid_vec.inverse().apply_to_point(points)
  unit_vector = rigid_vec.normalized()
  unit_vector = [jax2th(unit_vector.x), jax2th(unit_vector.y), jax2th(unit_vector.z)]

  return unit_vector, jax2th(rigid_mask)


def cross(e0, e1):
  """
  Compute cross product between e0 and e1.
  # e0: [num_res, 3]
  # e1: [num_res, 3]
  # out: [num_res,]
  """
  # e2 = torch.zeros_like(e0)
  # e2[:, 0] = e0[:, 1] * e1[:, 2] - e0[:, 2] * e1[:, 1]
  # e2[:, 1] = e0[:, 2] * e1[:, 0] - e0[:, 0] * e1[:, 2]
  # e2[:, 2] = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]

  return torch.cross(e0, e1, dim=-1)

def dot(e0, e1):
  """Compute dot product between 'self' and 'other'."""
  # e0: [num_res, 3]
  # e1: [num_res, 3]
  # out: [num_res,]
  # return self.x * other.x + self.y * other.y + self.z * other.z
  return (e0 * e1).sum(-1)

def norm(e, epsilon: float = 1e-6):
  """Compute Norm of Vec3Array, clipped to epsilon."""
  # e: [num_res, 3]
  # To avoid NaN on the backward pass, we must use maximum before the sqrt
  norm2 = dot(e, e) # [num_res,]
  if epsilon:
    norm2 = torch.maximum(norm2, torch.tensor(epsilon**2, dtype=norm2.dtype))
  return torch.sqrt(norm2) # [num_res,]

def normalized(e, epsilon: float = 1e-6):
  """
  Return unit vector with optional clipping.
  e: [num_res, 3]
  out: [num_res, 3]
  """
  return e / norm(e, epsilon).unsqueeze(-1)

def from_two_vectors(e0, e1):
  e0 = normalized(e0)  # [num_res, 3] 
  # make e1 perpendicular to e0.
  c = dot(e1, e0)  # [num_res] 
  e1 = normalized(e1 - (c[:, None] * e0)) # [num_res, 3] 
  # Compute e2 as cross product of e0 and e1.
  e2 = cross(e0, e1) # [num_res, 3] 

  rotation = torch.stack((e0, e1, e2), dim=-1) # [num_res, 3, 3] 
  # (e0[:, 0], e1[:, 0], e2[:, 0], 
  #  e0[:, 1], e1[:, 1], e2[:, 1], 
  #  e0[:, 2], e1[:, 2], e2[:, 2])
  """
  (
  xx,xy,xz
  yx,yy,yz
  zx,zy,zz
  )
  """
  return rotation # Tuple[num_res,]

def rot_apply_to_point(rotation, point):
  """Applies Rot3Array to point."""
  """
  rotation: [num_res, 3, 3] 
  point: [num_res, 3] 
  return vector.Vec3Array(
      self.xx * point.x + self.xy * point.y + self.xz * point.z,
      self.yx * point.x + self.yy * point.y + self.yz * point.z,
      self.zx * point.x + self.zy * point.y + self.zz * point.z,
  )
  """
  point_out = (rotation * point.unsqueeze(-2)).sum(-1) # [num_res, 3] 
  return point_out

def apply_to_point(rotation, translation, point):
  """Apply Rigid3Array transform to point."""
  return rot_apply_to_point(rotation, point) + translation

def inverse_rot(rotation):
  """Returns inverse of Rot3Array."""
  inverse = rotation.transpose(-1, -2)
  """
  return Rot3Array(
      *(self.xx, self.yx, self.zx),
      *(self.xy, self.yy, self.zy),
      *(self.xz, self.yz, self.zz),
  )
  """
  return inverse
  

def inverse(rotation, translation):
  """Return Rigid3Array corresponding to inverse transform."""
  inv_rotation = inverse_rot(rotation)
  inv_translation = rot_apply_to_point(inv_rotation, -translation)
  return inv_rotation, inv_translation

def make_backbone_vectors_th(
    positions: torch.Tensor,
    mask: torch.Tensor,
    group_indices: torch.Tensor,
):
    """PyTorch 实现的蛋白质骨架刚性变换生成
    
    Args:
        positions: [num_res, num_atoms, 3] 原子位置 (Vec3Array)
        mask: [num_res, num_atoms] 原子掩码
        group_indices: [num_res, num_group, 3] 原子索引组
    
    Returns:
        (Rot3Array, [num_res] 掩码)
    """
    batch_index = torch.arange(mask.shape[0], dtype=torch.long)

    # 提取主干原子索引 (N, CA, C)
    backbone_indices = group_indices[:, 0].long()  # [num_res, 3]
    # print(positions.shape)
    # print(backbone_indices.shape)
    c, b, a = torch.unbind(backbone_indices, -1)
    # print(a.shape)

    # 计算刚性变换掩码 (所有三个原子必须有效)
    slice_index = lambda x, idx: x[batch_index, idx]
    rigid_mask = (
        slice_index(mask, a) * slice_index(mask, b) * slice_index(mask, c)
    ).float()  # [num_res]
    # print(rigid_mask.shape)

    frame_positions = []
    # positions  [num_res, num_atoms, 3] 
    for indices in [a, b, c]:
      frame_positions.append(
        slice_index(positions, indices)
    )
      
    # frame_positions  List[num_res, num_atoms, 3] 

    # 构建旋转矩阵 (从 CA->C 和 CA->N 向量)
    e0 = frame_positions[2] - frame_positions[1]  # [num_res, 3] 
    e1 = frame_positions[0] - frame_positions[1]  # [num_res, 3] 

    rotation = from_two_vectors(e0, e1)
    translation = frame_positions[1]
    rigid = (rotation[:, None, :, :], translation[:, None, :])

    points = translation
    rigid_vec = apply_to_point(*inverse(*rigid), points) # [num_res, 3] 
    unit_vector = normalized(rigid_vec)
    unit_vector = torch.unbind(unit_vector, -1)

    return unit_vector, rigid_mask