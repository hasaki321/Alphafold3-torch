import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numbers
from typing import Sequence, TypeAlias, Optional, Tuple, Union
import intel_extension_for_pytorch as ipex

PRECISION: TypeAlias = Optional[Union[str, torch.dtype, Tuple[str, str], Tuple[torch.dtype, torch.dtype]]]

DEFAULT_PRECISION = None

TRUNCATED_NORMAL_STDDEV_FACTOR = np.asarray(
    0.87962566103423978, dtype=np.float32
)

layernorm = F.layer_norm

class LayerNorm(nn.Module):
  """LayerNorm module, equivalent to hk.LayerNorm but in PyTorch."""

  def __init__(
      self,
      normalized_shape: int, # Changed axis to normalized_shape for clarity in PyTorch
      *,
      eps: float = 1e-5,
      create_scale: bool = True,
      create_offset: bool = True,
      use_fast_variance: bool = True, # Not directly applicable, can be ignored or customized in forward
      name: str = None, # name is not directly used in forward, but kept for potential future use
      param_axis: Optional[int] = None, # Not directly applicable, assuming last dim in PyTorch
      upcast: bool = True,
  ):
    super().__init__()
    self.eps = eps
    self.upcast = upcast
    self.normalized_shape = normalized_shape #  Storing normalized_shape
    # print(f'using {self.layernorm} ln')

    self.weight = None
    self.bias = None

    if create_scale:
        self.weight = nn.Parameter(torch.ones(normalized_shape)) # Correct shape
    if create_offset:
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) # Correct shape


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    is_16bit = x.dtype in [torch.float16, torch.bfloat16] # Correct dtype check for PyTorch
    if self.upcast and is_16bit:
      x = x.float() # Upcast to float32

    scale = self.weight if hasattr(self, 'weight') and self.weight is not None else None # Handle cases where scale/offset might not be created
    offset = self.bias if hasattr(self, 'bias') and self.bias is not None else None
    # F.layer_norm
    out = layernorm(
        x,
        normalized_shape=(self.normalized_shape,), # Pass normalized_shape as tuple
        weight=scale,
        bias=offset,
        eps=self.eps
    )

    if self.upcast and is_16bit:
      out = out.to(dtype) # Cast back to original dtype

    return out


def torch_linear_get_params(
    inputs: torch.Tensor, # Changed inputs to torch.Tensor
    num_output: Union[int, Sequence[int]],
    use_bias: bool = False,
    num_input_dims: int = 1,
    initializer: str = 'linear',
    bias_init: float = 0.0,
    transpose_weights: bool = False,
    name: str = None, # name is not used, but kept for consistency
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
  """Get parameters for linear layer in Torch style."""

  if isinstance(num_output, numbers.Integral):
    output_shape = (num_output,)
  else:
    output_shape = tuple(num_output)

  if num_input_dims > 0:
    in_shape = inputs.shape[-num_input_dims:]
  elif num_input_dims == 0:
    in_shape = ()
  else:
    raise ValueError('num_input_dims must be >= 0.')

  weight_init = _get_initializer_scale(initializer, in_shape)

  if transpose_weights:
    weight_shape = output_shape + in_shape
  else:
    weight_shape = in_shape + output_shape

  weights = nn.Parameter(torch.empty(weight_shape, dtype=inputs.dtype)) # Create parameter, init later
  if transpose_weights:
      weights.data = weight_init(weights.data) # Apply initializer, assuming it returns tensor
  else:
      weights.data = weight_init(weights.data.T).T # Apply initializer and transpose back


  bias = None
  if use_bias:
    bias = nn.Parameter(torch.empty(output_shape, dtype=inputs.dtype)) # Create bias parameter
    nn.init.constant_(bias, bias_init) # Use torch constant init

  return weights, bias


class Linear(nn.Module):
  """Custom Linear Module in PyTorch."""

  def __init__(
      self,
      num_output: Union[int, Sequence[int]],
      *,
      initializer: str = 'linear',
      num_input_dims: int = 1,
      use_bias: bool = False,
      bias_init: float = 0.0,
      precision: PRECISION = None,
      fast_scalar_mode: bool = True,
      transpose_weights: bool = False,
      name: str = None,
  ):
    super().__init__()
    if isinstance(num_output, numbers.Integral):
      self.output_shape = (num_output,)
    else:
      self.output_shape = tuple(num_output)
    self.initializer = initializer
    self.use_bias = use_bias
    self.bias_init = bias_init
    self.num_input_dims = num_input_dims
    self.num_output_dims = len(self.output_shape)
    self.precision = precision # Precision handling might need more customization for torch
    self.fast_scalar_mode = fast_scalar_mode
    self.transpose_weights = transpose_weights

    self.weight = None
    self.bias = None


  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Connects Module."""

    num_input_dims = self.num_input_dims

    if num_input_dims == 0 and self.fast_scalar_mode:
      weight_shape = self.output_shape
      self.weight = nn.Parameter(torch.empty(weight_shape, dtype=inputs.dtype))
      if self.initializer == 'zeros':
        nn.init.constant_(self.weight, 0.0)
      else:
        distribution_stddev = torch.tensor(1 / TRUNCATED_NORMAL_STDDEV_FACTOR).float() # Torch tensor for stddev
        init_func = get_truncated_normal_initializer(mean=0.0, std=distribution_stddev) # Use custom truncated normal init
        self.weight.data = init_func(self.weight.data)


      inputs = inputs.view(*inputs.shape + (1,) * self.num_output_dims) # Expand dims using view
      output = inputs * self.weight
    else:
      if self.num_input_dims > 0:
        in_shape = inputs.shape[-self.num_input_dims :]
      else:
        in_shape = ()

      weight_init = _get_initializer_scale(self.initializer, in_shape)

      in_letters = 'abcde'[: self.num_input_dims]
      out_letters = 'hijkl'[: self.num_output_dims]

      if self.transpose_weights:
        weight_shape = self.output_shape + in_shape
      else:
        weight_shape = in_shape + self.output_shape

      self.weight = nn.Parameter(torch.empty(weight_shape, dtype=inputs.dtype))
      if self.transpose_weights:
          self.weight.data = weight_init(self.weight.data)
      else:
          self.weight.data = weight_init(self.weight.data.T).T


      equation = (
          f'...{in_letters}, {in_letters}{out_letters}->...{out_letters}'
      ) if not self.transpose_weights else  (
          f'...{in_letters}, {out_letters}{in_letters}->...{out_letters}'
      )

      output = torch.einsum(equation, inputs, self.weight) # Use torch.einsum

    if self.use_bias:
      self.bias = nn.Parameter(torch.empty(self.output_shape, dtype=inputs.dtype))
      nn.init.constant_(self.bias, self.bias_init)
      output += self.bias

    return output


def _get_initializer_scale(initializer_name, input_shape):
  """Get initializer for weights, returns PyTorch initializer function."""

  if initializer_name == 'zeros':
    def init_func(tensor):
      return nn.init.constant_(tensor, 0.0)
    return init_func
  else:
    # fan-in scaling
    noise_scale = 1.0
    for channel_dim in input_shape:
      noise_scale /= channel_dim
    if initializer_name == 'relu':
      noise_scale *= 2

    stddev = np.sqrt(noise_scale)
    # Adjust stddev for truncation.
    stddev_truncated = stddev / TRUNCATED_NORMAL_STDDEV_FACTOR
    stddev_tensor = torch.tensor(stddev_truncated).float() # Convert to torch tensor

    def init_func(tensor):
      return get_truncated_normal_initializer(mean=0.0, std=stddev_tensor)(tensor) # Use truncated normal initializer
    return init_func


def get_truncated_normal_initializer(mean, std):
    """Returns a Truncated Normal initializer in PyTorch."""
    from scipy.stats import truncnorm

    def truncated_normal_(tensor):
        values = truncnorm.rvs(-2, 2, loc=mean, scale=std, size=tensor.shape) # Using scipy for truncation
        return torch.from_numpy(np.asarray(values)).float() # Convert to torch

    return truncated_normal_



if __name__ == "__main__":
    def test_layer_norm():
        print("Testing LayerNorm...")
        normalized_shape = 64
        layer_norm = LayerNorm(normalized_shape=normalized_shape, name='test_ln')

        # 打印 LayerNorm 的参数 (可选，用于查看初始化)
        print("LayerNorm parameters:")
        for name, param in layer_norm.named_parameters():
            print(f"  {name}: shape={param.shape}")

        input_tensor = torch.randn(2, 32, normalized_shape) # 示例输入形状
        output_tensor = layer_norm(input_tensor)

        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output_tensor.shape}")
        assert output_tensor.shape == input_tensor.shape, "LayerNorm output shape mismatch!"
        print("LayerNorm test passed!\n")


    def test_linear():
        print("Testing Linear...")
        num_output = 128
        num_input_dims = 1
        linear_layer = Linear(num_output=num_output, num_input_dims=num_input_dims, name='test_linear')

        # 打印 Linear 的参数 (可选，用于查看初始化)
        print("Linear parameters:")
        for name, param in linear_layer.named_parameters():
            print(f"  {name}: shape={param.shape}")

        input_tensor = torch.randn(4, 64) # 示例输入形状，最后一个维度是 input_dim
        output_tensor = linear_layer(input_tensor)

        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output_tensor.shape}")
        assert output_tensor.shape == torch.Size([4, num_output]), "Linear output shape mismatch!"

        # 检查权重参数的形状 (可选)
        weight_shape_expected = torch.Size([64, 128]) # 根据 num_input_dims 和 num_output
        assert linear_layer.weight.shape == weight_shape_expected, f"Linear weight shape mismatch! Expected {weight_shape_expected}, got {linear_layer.weight.shape}"

        print("Linear test passed!\n")


    def test_linear_scalar_input():
        print("Testing Linear with scalar input (fast_scalar_mode=True)...")
        num_output = (32, 16) # 多维输出
        linear_layer = Linear(num_output=num_output, num_input_dims=0, fast_scalar_mode=True, name='test_linear_scalar')

        # 打印 Linear (scalar input) 的参数
        print("Linear (scalar input) parameters:")
        for name, param in linear_layer.named_parameters():
            print(f"  {name}: shape={param.shape}")

        input_tensor = torch.randn(8) # 标量输入
        output_tensor = linear_layer(input_tensor)

        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output_tensor.shape}")
        assert output_tensor.shape == torch.Size([8, 32, 16]), "Linear (scalar input) output shape mismatch!"

        weight_shape_expected = torch.Size([32, 16]) # 输出形状
        assert linear_layer.weight.shape == weight_shape_expected, f"Linear (scalar input) weight shape mismatch! Expected {weight_shape_expected}, got {linear_layer.weight.shape}"


        print("Linear scalar input test passed!\n")


    def run_tests():
        test_layer_norm()
        test_linear()
        test_linear_scalar_input()
        print("All tests completed!")


    run_tests()