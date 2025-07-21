import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import jit, device_put, device_get
import gc
from functools import partial

import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

from openpi.shared.download import maybe_download
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config
from openpi.models.model import restore_params

from safetensors.numpy import save_file
import shutil
from orbax.checkpoint import PyTreeCheckpointer
from openpi.training.checkpoints import initialize_checkpoint_dir

TEST_MODEL = False 
VIEW_MODEL_ARCHITECTURE = False 
PRUNE_MODEL = True
PRUNING_ALGORITHM = 'Magnitude' # or 'Wanda'

KEYS_TO_PRUNE = [
  # pi0‐image‐Transformer MLP kernels
  'PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel',
  'PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel',

  # pi0‐image‐Transformer attention kernels
  'PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel',
  'PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel',
  'PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel',
  'PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel',

  # vision‐backbone conv→linear projection kernels
  'PaliGemma/img/embedding/kernel',
  'PaliGemma/img/head/kernel',

  # Gemma LLM “einsum” weights  
  'PaliGemma/llm/layers/attn/q_einsum/w',
  'PaliGemma/llm/layers/attn/kv_einsum/w',
  'PaliGemma/llm/layers/attn/attn_vec_einsum/w',

  # Gemma FFN gating + output  
  'PaliGemma/llm/layers/mlp/gating_einsum',
  'PaliGemma/llm/layers/mlp/linear',
]


def view_model_architecture(local_ckpt):

    # Load the raw params PyTree from the checkpoint
    params = restore_params(local_ckpt / "params", restore_type=np.ndarray)

    # Flatten the dict so each key is a tuple path of strings
    flat_params = flatten_dict(params, sep="/")

    # total = 0
    # for path, arr in flat_params.items():
    #     size = arr.size
    #     total += size
    #     print(f"{path:70} │ shape={arr.shape!s:15} │ params={size:,}")

    total = 0
    scan_keys = ("encoderblock", "layers")  # any path containing these we’ll unroll
    for path, arr in flat_params.items():
        # detect scanned modules: those with a first‐axis = num_layers
        if arr.ndim >= 3 and any(k in path for k in scan_keys):
            # unroll along axis 0
            for layer_idx, layer_arr in enumerate(arr):
                size = layer_arr.size
                total += size
                print(
                    f"{path}[{layer_idx:2d}]".ljust(75),
                    "│",
                    f"shape={tuple(layer_arr.shape)!s:15}",
                    "│",
                    f"params={size:,}"
                )
        else:
            # normal (un‐scanned) parameters
            size = arr.size
            total += size
            print(
                path.ljust(75),
                "│",
                f"shape={tuple(arr.shape)!s:15}",
                "│",
                f"params={size:,}"
            )

    print(f"\n Total parameters in model: {total:,}")


def test_model_output(model_name, local_ckpt):
    # 3) Build the TrainConfig and then the Policy object:
    train_cfg = get_config(model_name)
    policy    = create_trained_policy(train_cfg, local_ckpt)


    # Example: prepare a dummy observation dict
    dummy_obs = {
        "observation/image":      np.zeros((224,224,3), dtype=np.uint8),
        "observation/wrist_image":np.zeros((224,224,3), dtype=np.uint8),
        "observation/state":      np.zeros((8,),       dtype=np.float32),
        "prompt":                 "place the red block on the green target",
    }

    # 4) Run a single inference step:
    action_chunk = policy.infer(dummy_obs)["actions"]
    print("Sampled actions:", action_chunk)


@partial(jit, donate_argnums=(0,))
def _magnitude_prune_layer_jax(x: jnp.ndarray) -> jnp.ndarray:
    keep, block = 2, 4
    *rest, last = x.shape
    n_blocks = last // block
    limit    = n_blocks * block

    blocks   = x[..., :limit].reshape(*rest, n_blocks, block)
    abs_b    = jnp.abs(blocks)
    thresh   = jnp.partition(abs_b, -keep, axis=-1)[..., -keep]
    mask     = abs_b >= thresh[..., None]
    pruned   = blocks * mask

    return x.at[..., :limit].set(pruned.reshape(*rest, limit))

def magnitude_prune_layer_jax(arr: np.ndarray) -> np.ndarray:
    # Move to device
    gx = device_put(arr)

    # Prune in-place on GPU (input buffer is donated, so no extra copy)
    gx_pruned = _magnitude_prune_layer_jax(gx)

    # Force it to finish, copy back to host
    gx_pruned.block_until_ready()
    out = np.array(device_get(gx_pruned))

    # Now delete all GPU buffers & run Python GC
    del gx, gx_pruned
    gc.collect()

    return out


def magnitude_prune(local_ckpt, model_name="pi0_fast_libero", out_path="pi0_fast_libero_Magnitude_pruned"):

    params = restore_params(local_ckpt / "params", restore_type=np.ndarray)

    # flatten
    flat = flatten_dict(params, sep="/")

    total_params_pruned = 0

    # these are the blobs that come in shape (num_layers, …)
    SCAN_KEYS = ['gating_einsum', 'linear'] # ["attn_vec_einsum", "kv_einsum", "q_einsum", "gating_einsum", "linear"]

    for key in KEYS_TO_PRUNE:
        W = flat[key]
        N_before = W.size

        # sanity print
        print(f"\n— {key}  shape={W.shape}  total={N_before:,}")
        print(" before slice:\n", W.reshape(-1, W.shape[-1])[:8, :8])

        if any(k in key for k in SCAN_KEYS) and W.ndim >= 3:
            # fused across the first axis
            num_layers, *rest = W.shape
            for layer_idx in range(num_layers):
                sub = W[layer_idx]  # e.g. shape=(2,2048,16384)
                W_p_l = magnitude_prune_layer_jax(sub)
                W[layer_idx] = W_p_l
        else:
            # non‐scanned: just prune the whole matrix at once (in chunks)
            W = magnitude_prune_layer_jax(W)

        flat[key] = W

        print(" after slice:\n", W.reshape(-1, W.shape[-1])[:8, :8])

        total_params_pruned += N_before

    print(f"\n Total pruned weights in model: {total_params_pruned}")

    save_pruned_checkpoint(local_ckpt, flat, out_ckpt)


def wanda_prune(local_ckpt, model_name="pi0_fast_libero", out_path="pi0_fast_libero_Wanda_pruned"):

    params = restore_params(local_ckpt / "params", restore_type=np.ndarray)

    # flatten
    flat = flatten_dict(params, sep="/")

    total_params_pruned = 0

    # these are the blobs that come in shape (num_layers, …)
    SCAN_KEYS = ['gating_einsum', 'linear'] # ["attn_vec_einsum", "kv_einsum", "q_einsum", "gating_einsum", "linear"]


    # Calibrate activation norms
    print("Calibrating activation norms for layers:", KEYS_TO_PRUNE)
    act_norms = calibrate_activation_norms(model, params, cfg, KEYS_TO_PRUNE, num_steps=50)



def save_pruned_checkpoint(
    orig_ckpt: Path,
    pruned_flat: dict[str, np.ndarray],
    out_ckpt: Path,
):
    """
    orig_ckpt/
      assets/…
      params/
        _METADATA
        _sharding
        <zarr shards…>
    -->
    out_ckpt/   # new directory
      assets/…        copied verbatim
      params/         rewritten with pruned params
    """
    # unflatten to the same pytree structure
    pruned_pytree = {"params": unflatten_dict(pruned_flat, sep="/")}

    # copy over the assets dir so nothing else in your pipeline breaks
    shutil.copytree(orig_ckpt / "assets", out_ckpt / "assets", dirs_exist_ok=True)

    # make a fresh params directory
    params_dir = out_ckpt / "params"
    if params_dir.exists():
        shutil.rmtree(params_dir)
    # params_dir.mkdir(parents=True, exist_ok=True)

    # use Orbax’s PyTreeCheckpointer to write exactly the same zarr layout
    ckptr = PyTreeCheckpointer()
    ckptr.save(params_dir,pruned_pytree)
    print(f"Wrote pruned checkpoint to {out_ckpt}")

if __name__ == "__main__":

    model_name    = "pi0_fast_libero"  # or your pi0 checkpoint name
    checkpoint_uri = f"gs://openpi-assets/checkpoints/{model_name}"
    local_ckpt = Path(maybe_download(checkpoint_uri))
    out_ckpt  = Path(f"/home/ubuntu/vla/models/pi0_fast_libero_{PRUNING_ALGORITHM}_pruned_orbax")
    print(f"Using checkpoint at: {local_ckpt}")

    if TEST_MODEL:
        test_model_output(model_name, local_ckpt)
    
    if VIEW_MODEL_ARCHITECTURE:
        view_model_architecture(local_ckpt)

    if PRUNE_MODEL:
        if PRUNING_ALGORITHM == 'Magnitude':
            magnitude_prune(local_ckpt, model_name=model_name, out_path=out_ckpt)
        elif PRUNING_ALGORITHM == 'Wanda':
            raise NotImplementedError("Wanda pruning is not implemented yet.")
        else:
            raise ValueError(f"Unknown pruning algorithm: {PRUNING_ALGORITHM}")