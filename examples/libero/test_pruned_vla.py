import numpy as np

from openpi import transforms
from openpi.training.config import get_config
from openpi.policies.policy_config import create_trained_policy
from openpi.training.data_loader import create_data_loader
from openpi.transforms import Group, RepackTransform
from dataclasses import asdict
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main():
  model_name  = "pi0_fast_libero"
  pruned_path = "/home/ubuntu/vla/models/pi0_fast_libero_Magnitude_pruned_orbax"
  # pruned_path = "/home/ubuntu/.cache/openpi/openpi-assets/checkpoints/pi0_fast_libero"
  cfg    = get_config(model_name)
  policy = create_trained_policy(cfg, pruned_path)

  # Load the converted dataset
  dataset = LeRobotDataset(repo_id=cfg.data.repo_id)  

  dataset_iter = iter(dataset)
  for _ in range(1000):
    sample = next(dataset_iter) 

    # build the correctly-namespaced input dict
    obs = {
        "observation/image":       sample["image"],
        "observation/wrist_image": sample["wrist_image"],
        "observation/state":       sample["state"],
        "prompt":                  sample.get("task", None),
    }

    # Perform inference
    outputs = policy.infer(obs)
    # outputs["actions"] has shape (7,) for the 7-DOF Libero action
    print("Predicted actions:", outputs["actions"])


  # ############################################################

  # ## Now let's run a few episodes of inference on the dataset

  # ############################################################
  # for e_idx in range(3):

  #   # Get the episode start and end indices
  #   from_idx = dataset.episode_data_index["from"][e_idx].item()
  #   to_idx   = dataset.episode_data_index["to"][e_idx].item()        

  #   # Slice out that episode
  #   episode_steps = [ dataset[i] for i in range(from_idx, to_idx + 1) ]

  #   # Iterate through each step for inference
  #   for step in episode_steps:
  #       obs = {
  #           "observation/image":       sample["image"],
  #           "observation/wrist_image": sample["wrist_image"],
  #           "observation/state":       sample["state"],
  #           "prompt":                  sample.get("task", None),
  #       }
  #       actions = policy.infer(obs)["actions"]
  #       print(f"Episode {e_idx} Frame {step['frame_index'].item()}: actions = {actions}")



if __name__ == "__main__":
  main()