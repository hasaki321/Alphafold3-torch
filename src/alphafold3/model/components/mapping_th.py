import torch
import functools
from collections.abc import Callable, Sequence
from typing import Any

Pytree = Any
PytreeTorchTensor = Any

partial = functools.partial
PROXY = object()


def _maybe_slice(array, i, slice_size, axis):
    if axis is PROXY:
        return array
    else:
        return array.narrow(dim=axis, start=i, length=slice_size)


def _maybe_get_size(array, axis):
    if axis == PROXY:
        return -1
    else:
        return array.shape[axis]


def _expand_axes(axes, values, name="sharded_apply"):
    # In PyTorch, we can't directly replicate tree structure handling like in JAX,
    # so we flatten the structure manually if needed.
    if isinstance(values, torch.Tensor):
        return [axes]
    else:
        return [None]  # This needs adaptation for non-tensor structures.
from typing import Dict

def sharded_map(
    fun: Callable[..., PytreeTorchTensor],
    shard_size: int | None = 1,
    in_axes: int | Pytree = 0,
    out_axes: int | Pytree = 0,
) -> Callable[..., PytreeTorchTensor]:
    """Sharded vmap equivalent in PyTorch."""

    def mapped_fn(*args, **kwargs):
        if shard_size is None:
            return fun(*args, **kwargs)

        # Expand in axes and determine loop range.
        in_axes_ = _expand_axes(in_axes, args)

        in_sizes = [x.shape[0] for x in args]  # Assuming axis 0 is the batching axis
        in_size = max(in_sizes)

        num_extra_shards = (in_size - 1) // shard_size
        last_shard_size = in_size % shard_size
        last_shard_size = shard_size if last_shard_size == 0 else last_shard_size

        def apply_fun_to_slice(slice_start, slice_size):
            # Slice the inputs
            input_slice = [x.narrow(0, slice_start, slice_size) for x in args]
            return fun(*input_slice, **kwargs)

        outputs = []
        for slice_start in range(0, in_size, shard_size):
            slice_size = shard_size if slice_start + shard_size <= in_size else last_shard_size
            output = apply_fun_to_slice(slice_start, slice_size)
            outputs.append(output)
        output_final = None
        if isinstance(outputs[0],torch.Tensor):
            output_final = torch.cat(outputs, dim=0)
        elif isinstance(outputs[0],Dict):
        # for confidence head
            output_final = {
                # k: torch.cat([output[k] if output[k].dim()!=0 else torch.unsqueeze(output[k],0) for output in outputs], dim=0)
                k: torch.stack([output[k] for output in outputs], dim=0)
                for k in outputs[0]
            }
        return output_final

    return mapped_fn


def reshape_partitioned_inputs(
    batched_args: Sequence[PytreeTorchTensor],
    partitioned_dim: int,
    subbatch_size: int,
) -> Sequence[PytreeTorchTensor]:
    """Reshapes inputs so subbatching doesn't happen on the partitioned dim."""
    subbatched_args = []
    for arg in batched_args:
        shape = arg.shape
        new_shape = (
            shape[:partitioned_dim]
            + (subbatch_size, shape[partitioned_dim] // subbatch_size)
            + shape[partitioned_dim + 1 :]
        )
        subbatched_args.append(arg.view(new_shape))
    return subbatched_args


def reshape_partitioned_output(output: torch.Tensor, output_subbatch_dim: int) -> torch.Tensor:
    """Reshapes outputs as if reshape_partitioned_inputs were never applied."""
    out_shape = (
        output.shape[: output_subbatch_dim - 1]
        + (-1,)
        + output.shape[output_subbatch_dim + 1 :]
    )
    return output.view(out_shape)


def inference_subbatch(
    module: Callable[..., PytreeTorchTensor],
    subbatch_size: int,
    batched_args: Sequence[PytreeTorchTensor],
    nonbatched_args: Sequence[PytreeTorchTensor],
    input_subbatch_dim: int = 0,
    output_subbatch_dim: int | None = None,
    input_subbatch_dim_is_partitioned: bool = False,
) -> PytreeTorchTensor:
    """Run through subbatches (like batch apply but with split and concat)."""
    assert len(batched_args) > 0  # pylint: disable=g-explicit-length-test

    if output_subbatch_dim is None:
        output_subbatch_dim = input_subbatch_dim

    if input_subbatch_dim_is_partitioned:
        batched_args = reshape_partitioned_inputs(
            batched_args, input_subbatch_dim, subbatch_size
        )
        input_subbatch_dim += 1
        output_subbatch_dim += 1
        subbatch_size = 1

    def run_module(*batched_args):
        if input_subbatch_dim_is_partitioned:
            batched_args = [b.squeeze(dim=input_subbatch_dim) for b in batched_args]
        args = list(batched_args) + list(nonbatched_args)
        res = module(*args)
        if input_subbatch_dim_is_partitioned:
            res = res.unsqueeze(dim=output_subbatch_dim)
        return res

    sharded_module = sharded_map(
        run_module,
        shard_size=subbatch_size,
        in_axes=input_subbatch_dim,
        out_axes=output_subbatch_dim,
    )
    output = sharded_module(*batched_args)
    if input_subbatch_dim_is_partitioned:
        output = reshape_partitioned_output(output, output_subbatch_dim)

    return output
