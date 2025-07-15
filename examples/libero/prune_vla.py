import numpy as np
from pathlib import Path
import sys

from openpi.shared.download import maybe_download
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config

def load_libero_policy():
    # 1) Identify which config & checkpoint to use:
    model_name    = "pi0_fast_libero"
    checkpoint_uri = f"gs://openpi-assets/checkpoints/{model_name}"

    # 2) Download (or reuse) the checkpoint locally:
    local_ckpt = maybe_download(checkpoint_uri)
    print(f"Using checkpoint at: {local_ckpt}")

    # 3) Build the TrainConfig and then the Policy object:
    train_cfg = get_config(model_name)
    policy    = create_trained_policy(train_cfg, local_ckpt)

    print("The policy is:", policy)
    print("The policy type is: ", type(policy))


    return policy

if __name__ == "__main__":
    policy = load_libero_policy()

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