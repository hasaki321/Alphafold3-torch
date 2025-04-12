# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Model param loading."""

import bisect
import collections
from collections.abc import Iterator
import contextlib
import io
import os
import pathlib
import re
import struct
import sys
from typing import IO

import haiku as hk
import jax.numpy as jnp
import numpy as np
import zstandard


class RecordError(Exception):
  """Error reading a record."""


def encode_record(scope: str, name: str, arr: np.ndarray) -> bytes:
  """Encodes a single haiku param as bytes, preserving non-numpy dtypes."""
  scope = scope.encode('utf-8')
  name = name.encode('utf-8')
  shape = arr.shape
  dtype = str(arr.dtype).encode('utf-8')
  arr = np.ascontiguousarray(arr)
  if sys.byteorder == 'big':
    arr = arr.byteswap()
  arr_buffer = arr.tobytes('C')
  header = struct.pack(
      '<5i', len(scope), len(name), len(dtype), len(shape), len(arr_buffer)
  )
  return header + b''.join(
      (scope, name, dtype, struct.pack(f'{len(shape)}i', *shape), arr_buffer)
  )


def _read_record(stream: IO[bytes]) -> tuple[str, str, np.ndarray] | None:
  """Reads a record encoded by `_encode_record` from a byte stream."""
  header_size = struct.calcsize('<5i')
  header = stream.read(header_size)
  if not header:
    return None
  if len(header) < header_size:
    raise RecordError(f'Incomplete header: {len(header)=} < {header_size=}')
  (scope_len, name_len, dtype_len, shape_len, arr_buffer_len) = struct.unpack(
      '<5i', header
  )
  fmt = f'<{scope_len}s{name_len}s{dtype_len}s{shape_len}i'
  payload_size = struct.calcsize(fmt) + arr_buffer_len
  payload = stream.read(payload_size)
  if len(payload) < payload_size:
    raise RecordError(f'Incomplete payload: {len(payload)=} < {payload_size=}')
  scope, name, dtype, *shape = struct.unpack_from(fmt, payload)
  scope = scope.decode('utf-8')
  name = name.decode('utf-8')
  dtype = dtype.decode('utf-8')
  arr = np.frombuffer(payload[-arr_buffer_len:], dtype=dtype)
  arr = np.reshape(arr, shape)
  if sys.byteorder == 'big':
    arr = arr.byteswap()
  return scope, name, arr


def read_records(stream: IO[bytes]) -> Iterator[tuple[str, str, np.ndarray]]:
  """Fully reads the contents of a byte stream."""
  while record := _read_record(stream):
    yield record


class _MultiFileIO(io.RawIOBase):
  """A file-like object that presents a concatenated view of multiple files."""

  def __init__(self, files: list[pathlib.Path]):
    self._files = files
    self._stack = contextlib.ExitStack()
    self._handles = [
        self._stack.enter_context(file.open('rb')) for file in files
    ]
    self._sizes = []
    for handle in self._handles:
      handle.seek(0, os.SEEK_END)
      self._sizes.append(handle.tell())
    self._length = sum(self._sizes)
    self._offsets = [0]
    for s in self._sizes[:-1]:
      self._offsets.append(self._offsets[-1] + s)
    self._abspos = 0
    self._relpos = (0, 0)

  def _abs_to_rel(self, pos: int) -> tuple[int, int]:
    idx = bisect.bisect_right(self._offsets, pos) - 1
    return idx, pos - self._offsets[idx]

  def close(self):
    self._stack.close()

  def closed(self) -> bool:
    return all(handle.closed for handle in self._handles)

  def fileno(self) -> int:
    return -1

  def readable(self) -> bool:
    return True

  def tell(self) -> int:
    return self._abspos

  def seek(self, pos: int, whence: int = os.SEEK_SET, /):
    match whence:
      case os.SEEK_SET:
        pass
      case os.SEEK_CUR:
        pos += self._abspos
      case os.SEEK_END:
        pos = self._length - pos
      case _:
        raise ValueError(f'Invalid whence: {whence}')
    self._abspos = pos
    self._relpos = self._abs_to_rel(pos)

  def readinto(self, b: bytearray | memoryview) -> int:
    result = 0
    mem = memoryview(b)
    while mem:
      self._handles[self._relpos[0]].seek(self._relpos[1])
      count = self._handles[self._relpos[0]].readinto(mem)
      result += count
      self._abspos += count
      self._relpos = self._abs_to_rel(self._abspos)
      mem = mem[count:]
      if self._abspos == self._length:
        break
    return result


@contextlib.contextmanager
def open_for_reading(model_files: list[pathlib.Path], is_compressed: bool):
  with contextlib.closing(_MultiFileIO(model_files)) as f:
    if is_compressed:
      yield zstandard.ZstdDecompressor().stream_reader(f)
    else:
      yield f


def _match_model(
    paths: list[pathlib.Path], pattern: re.Pattern[str]
) -> dict[str, list[pathlib.Path]]:
  """Match files in a directory with a pattern, and group by model name."""
  models = collections.defaultdict(list)
  for path in paths:
    match = pattern.fullmatch(path.name)
    if match:
      models[match.group('model_name')].append(path)
  return {k: sorted(v) for k, v in models.items()}


def select_model_files(
    model_dir: pathlib.Path, model_name: str | None = None
) -> tuple[list[pathlib.Path], bool]:
  """Select the model files from a model directory."""
  files = [file for file in model_dir.iterdir() if file.is_file()]

  for pattern, is_compressed in (
      (r'(?P<model_name>.*)\.[0-9]+\.bin\.zst$', True),
      (r'(?P<model_name>.*)\.bin\.zst\.[0-9]+$', True),
      (r'(?P<model_name>.*)\.[0-9]+\.bin$', False),
      (r'(?P<model_name>.*)\.bin]\.[0-9]+$', False),
      (r'(?P<model_name>.*)\.bin\.zst$', True),
      (r'(?P<model_name>.*)\.bin$', False),
  ):
    models = _match_model(files, re.compile(pattern))
    if model_name is not None:
      if model_name in models:
        return models[model_name], is_compressed
    else:
      if models:
        if len(models) > 1:
          raise RuntimeError(f'Multiple models matched in {model_dir}')
        _, model_files = models.popitem()
        return model_files, is_compressed
  raise FileNotFoundError(f'No models matched in {model_dir}')
import torch

def get_model_haiku_params(model_dir: pathlib.Path) -> hk.Params:
  """Get the Haiku parameters from a model name."""
  params: dict[str, dict[str, jnp.Array]] = {}
  model_files, is_compressed = select_model_files(model_dir)
  with open_for_reading(model_files, is_compressed) as stream:
    for scope, name, arr in read_records(stream):
      params.setdefault(scope, {})[name] = jnp.array(arr)
  if not params:
    raise FileNotFoundError(f'Model missing from "{model_dir}"')
  return params


from tqdm import tqdm
import torch
import jax
import json
def load_torch_params(model_dir:pathlib.Path, torch_2_jax_map_path:str = './models/map.json'):
    torch_2_jax_map = json.load(open(torch_2_jax_map_path,'r'))
    jax_params = get_model_haiku_params(model_dir)
    torch_state_dict = {}
    for torch_name, map_dict in torch_2_jax_map.items():
        jax_name = map_dict['jax_name']
        block_index_list = map_dict['block_index_list']
        torch_shape = map_dict['torch_shape']
        jax_arr = jax_params['/'.join(jax_name.split('/')[:-1])][jax_name.split('/')[-1]]

        # 步骤 1：将 JAX 数组转换为 DLPack 胶囊对象
        dlpack_capsule = jax.dlpack.to_dlpack(jax_arr)

        # 步骤 2：从 DLPack 胶囊对象创建 PyTorch Tensor，保留原始类型

        torch_tensor = torch.from_dlpack(dlpack_capsule)
        # jax_arr = np.array(jax_arr).astype(np.float32)
        # print(jax_name,jax_arr.dtype,jax_arr.shape, torch_tensor.dtype)
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
        ### confidence head  && diffusion head
        elif jax_name == 'diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/single_attention_q_projection/bias':
            assert len(block_index_list)==1
            torch_tensor = torch_tensor[block_index_list[0]].unsqueeze(0)
        elif jax_name == 'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderq_projection/bias':
            assert len(block_index_list)==1
            torch_tensor = torch_tensor[block_index_list[0]].unsqueeze(0)
        elif jax_name == 'diffuser/~/diffusion_head/diffusion_atom_transformer_encoder/__layer_stack_with_per_layer/diffusion_atom_transformer_encoderq_projection/bias':
            assert len(block_index_list)==1
            torch_tensor = torch_tensor[block_index_list[0]].unsqueeze(0)
        elif jax_name == 'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformerq_projection/bias':
            assert len(block_index_list)==2
            torch_tensor = torch_tensor[block_index_list[0]][block_index_list[1]].unsqueeze(0)

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
    return torch_state_dict
