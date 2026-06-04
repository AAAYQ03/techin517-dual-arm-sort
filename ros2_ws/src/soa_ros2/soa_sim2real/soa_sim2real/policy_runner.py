"""TorchScript policy loader and inference wrapper.

The exported ``policy.pt`` bakes in the ``EmpiricalNormalization`` module
(verified by inspecting the JIT graph), so callers pass raw 63-dim
observations and receive raw 6-dim action outputs.
"""

import numpy as np
import torch

from soa_sim2real.joint_order import OBS_DIM, ACTION_DIM


class PolicyRunner:
    def __init__(self, model_path: str, device: str = 'cuda', warmup_steps: int = 3):
        self._device = torch.device(device)
        self._model = torch.jit.load(model_path, map_location=self._device).eval()
        # Warmup so the first real call doesn't pay JIT specialization cost.
        with torch.inference_mode():
            zeros = torch.zeros((1, OBS_DIM), dtype=torch.float32, device=self._device)
            for _ in range(warmup_steps):
                self._model(zeros)

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """Run one forward pass.

        Args:
            obs: shape ``(OBS_DIM,)`` float32 array.
        Returns:
            shape ``(ACTION_DIM,)`` float32 array (raw, un-clamped).
        """
        if obs.shape != (OBS_DIM,):
            raise ValueError(f'expected obs shape ({OBS_DIM},), got {obs.shape}')

        with torch.inference_mode():
            obs_np = np.ascontiguousarray(obs, dtype=np.float32)
            obs_t = torch.from_numpy(obs_np).to(self._device).unsqueeze(0)  # (1, OBS_DIM)
            out = self._model(obs_t)
            return out.squeeze(0).cpu().numpy().astype(np.float32)
