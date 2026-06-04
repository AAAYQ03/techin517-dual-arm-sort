#!/usr/bin/env python3
"""测试 ACT load + 单次 inference"""
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.control_utils import predict_action

CHECKPOINT = "/home/ubuntu/techin517/outputs/train/act_battery_v1/checkpoints/last/pretrained_model"

print("Loading policy config ...")
policy_cfg = PreTrainedConfig.from_pretrained(CHECKPOINT)
print(f"  policy type: {type(policy_cfg).__name__}")
print(f"  device: {policy_cfg.device}")
print(f"  input_features: {list(policy_cfg.input_features.keys())}")
print(f"  output_features: {list(policy_cfg.output_features.keys())}")

print("\nLoading policy weights ...")
# make_policy 需要 ds_meta, 但实际从 pretrained_path 也能加载
from lerobot.policies.factory import get_policy_class
policy_class = get_policy_class(policy_cfg.type)
policy = policy_class.from_pretrained(CHECKPOINT)
policy.to(policy_cfg.device)
policy.eval()
print(f"  policy loaded: {type(policy).__name__}")

print("\nLoading processors ...")
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy_cfg,
    pretrained_path=CHECKPOINT,
)
print(f"  preprocessor: {type(preprocessor).__name__}")
print(f"  postprocessor: {type(postprocessor).__name__}")

print("\nCreating fake observation ...")
# 模拟 wrist 1280x720 RGB + top 1280x720 RGB + state 6-dim
obs = {
    "observation.images.wrist": np.zeros((720, 1280, 3), dtype=np.uint8),
    "observation.images.top": np.zeros((720, 1280, 3), dtype=np.uint8),
    "observation.state": np.zeros(6, dtype=np.float32),
}
print(f"  obs keys: {list(obs.keys())}")
print(f"  wrist shape: {obs['observation.images.wrist'].shape}")
print(f"  state shape: {obs['observation.state'].shape}")

print("\nRunning predict_action ...")
device = torch.device(policy_cfg.device)
action = predict_action(
    observation=obs,
    policy=policy,
    device=device,
    preprocessor=preprocessor,
    postprocessor=postprocessor,
    use_amp=False,
    task="Pick up the battery",
)
print(f"  action: {action}")
print(f"  action shape: {action.shape}")
print(f"  action numpy: {action.cpu().numpy() if torch.is_tensor(action) else action}")

print("\nOK")
