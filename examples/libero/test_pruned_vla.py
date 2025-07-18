import numpy as np

from openpi.training.config      import get_config
from openpi.policies.policy_config import create_trained_policy

def main()
  model_name  = "pi0_fast_libero"
  pruned_path = "/home/ubuntu/vla/models/pi0_fast_libero_Magnitude_pruned_orbax"

  cfg    = get_config(model_name)
  policy = create_trained_policy(cfg, pruned_path)

  dummy_obs = {
    "observation/image":       np.zeros((224,224,3), dtype=np.uint8),
    "observation/wrist_image": np.zeros((224,224,3), dtype=np.uint8),
    "observation/state":       np.zeros((8,),       dtype=np.float32),
    "prompt":                  "place the red block on the green target",
  }

  out = policy.infer(dummy_obs)
  print("Sampled actions:", out["actions"])

if __name__ == "__main__":
  main()