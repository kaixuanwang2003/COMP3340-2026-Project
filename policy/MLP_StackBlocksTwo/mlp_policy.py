import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
import pickle
import os


class StateMLP(nn.Module):
    """MLP policy that takes state-based observations for stack_blocks_two task"""

    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256, 256]):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Build MLP layers
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, obs):
        """Forward pass through the MLP"""
        return self.mlp(obs)


class MLPStackBlocksTwoPolicy:
    """MLP Policy for stack_blocks_two task using state-based observations"""

    def __init__(self, args_override=None):
        if args_override is None:
            args_override = {}

        # Define observation and action dimensions for stack_blocks_two
        # Proprioceptive: left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1) = 14
        # Block states: block1_pose(7) + block2_pose(7) = 14
        # Total obs_dim = 28
        self.obs_dim = args_override.get("obs_dim", 28)
        self.action_dim = args_override.get("action_dim", 14)  # 6 joints + 1 gripper per arm

        # Create the MLP model
        hidden_dims = args_override.get("hidden_dims", [256, 256, 256])
        self.model = StateMLP(self.obs_dim, self.action_dim, hidden_dims)

        # Device configuration
        self.device = torch.device(args_override.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Normalization stats
        self.obs_mean = None
        self.obs_std = None
        self.action_mean = None
        self.action_std = None

        # Load checkpoint if provided
        ckpt_dir = args_override.get("ckpt_dir", "")
        if ckpt_dir:
            self.load_checkpoint(ckpt_dir)

        self.model.eval()

        # Action scaling for safety
        self.action_scale = args_override.get("action_scale", 1.0)

    def load_checkpoint(self, ckpt_dir):
        """Load model weights and normalization stats"""
        # Load model weights
        ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded policy weights from {ckpt_path}")
        else:
            print(f"Warning: Could not find policy checkpoint at {ckpt_path}")

        # Load normalization stats
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                stats = pickle.load(f)
            self.obs_mean = torch.tensor(stats["obs_mean"], dtype=torch.float32, device=self.device)
            self.obs_std = torch.tensor(stats["obs_std"], dtype=torch.float32, device=self.device)
            self.action_mean = torch.tensor(stats["action_mean"], dtype=torch.float32, device=self.device)
            self.action_std = torch.tensor(stats["action_std"], dtype=torch.float32, device=self.device)
            print(f"Loaded normalization stats from {stats_path}")
        else:
            print(f"Warning: Could not find stats file at {stats_path}")

    def _normalize_obs(self, obs):
        """Normalize observations"""
        if self.obs_mean is not None and self.obs_std is not None:
            return (obs - self.obs_mean) / (self.obs_std + 1e-8)
        return obs

    def _denormalize_action(self, action):
        """Denormalize actions"""
        if self.action_mean is not None and self.action_std is not None:
            return action * self.action_std + self.action_mean
        return action

    def _extract_state_obs(self, obs_dict):
        """Extract state-based observations from the observation dictionary"""
        state_obs = []

        # Proprioceptive observations
        if "joint_action" in obs_dict:
            # Left arm joints (6) + left gripper (1)
            left_arm = obs_dict["joint_action"].get("left_arm", [0.0] * 6)
            left_gripper = obs_dict["joint_action"].get("left_gripper", 0.0)
            state_obs.extend(left_arm)
            state_obs.append(left_gripper)

            # Right arm joints (6) + right gripper (1)
            right_arm = obs_dict["joint_action"].get("right_arm", [0.0] * 6)
            right_gripper = obs_dict["joint_action"].get("right_gripper", 0.0)
            state_obs.extend(right_arm)
            state_obs.append(right_gripper)

        # Block states - try to get from observation or use defaults
        # In the stack_blocks_two task, we need to access block poses
        # For now, using placeholder - will be populated by the environment
        block1_pose = obs_dict.get("block1_pose", [0.0, 0.0, 0.741, 1.0, 0.0, 0.0, 0.0])  # x,y,z,qw,qx,qy,qz
        block2_pose = obs_dict.get("block2_pose", [0.0, 0.0, 0.741, 1.0, 0.0, 0.0, 0.0])

        state_obs.extend(block1_pose)
        state_obs.extend(block2_pose)

        return torch.tensor(state_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def predict_action(self, obs_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Predict action from observation dictionary"""
        # Extract state observations
        obs = self._extract_state_obs(obs_dict)

        # Normalize observations
        obs_norm = self._normalize_obs(obs)

        # Forward pass through MLP
        with torch.no_grad():
            action_norm = self.model(obs_norm)

        # Denormalize actions
        action = self._denormalize_action(action_norm)

        # Apply action scaling for safety
        action = action * self.action_scale

        # Convert to the expected format
        # Return as dictionary compatible with robotwin format
        return {
            "action": action.squeeze(0).cpu().numpy()
        }

    def reset(self):
        """Reset policy state (if any)"""
        pass

    def __call__(self, obs_dict):
        """Convenience method for prediction"""
        return self.predict_action(obs_dict)