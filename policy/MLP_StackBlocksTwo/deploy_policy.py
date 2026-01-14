import numpy as np
import torch
from .mlp_policy import MLPStackBlocksTwoPolicy
import yaml
import os


def encode_obs(observation):
    """
    Encode observation for MLP policy - extract state-based observations
    """
    obs_dict = {}

    # Proprioceptive observations (joint states)
    if "joint_action" in observation:
        obs_dict.update(observation["joint_action"])

    # Block poses - these need to be provided by the environment
    # In the stack_blocks_two task, we need access to block poses
    # This will be populated by the environment during evaluation
    if "block1_pose" in observation:
        obs_dict["block1_pose"] = observation["block1_pose"]
    if "block2_pose" in observation:
        obs_dict["block2_pose"] = observation["block2_pose"]

    return obs_dict


def get_model(usr_args):
    """
    Initialize and return the MLP policy model
    """
    # Default configuration
    model_config = {
        "obs_dim": 28,  # 14 (proprioceptive) + 14 (block states)
        "action_dim": 14,  # 6 joints + 1 gripper per arm
        "hidden_dims": [256, 256, 256],
        "device": usr_args.get("device", "cuda:0"),
        "action_scale": usr_args.get("action_scale", 1.0),
    }

    # Override with user arguments
    model_config.update(usr_args)

    # Set checkpoint directory
    if "ckpt_dir" not in model_config and "task_name" in usr_args:
        model_config["ckpt_dir"] = f"./policy/MLP_StackBlocksTwo/checkpoints/{usr_args['task_name']}"

    return MLPStackBlocksTwoPolicy(model_config)


def eval(TASK_ENV, model, observation):
    """
    Evaluate the MLP policy on the task environment

    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The MLP policy model from 'get_model()' function
    observation: The observation about the environment
    """
    # Encode observation for the policy
    obs = encode_obs(observation)

    # Add block poses to observation if available from the environment
    if hasattr(TASK_ENV, 'block1') and TASK_ENV.block1 is not None:
        block1_pose = TASK_ENV.block1.get_pose()
        obs["block1_pose"] = block1_pose.p.tolist() + block1_pose.q.tolist()

    if hasattr(TASK_ENV, 'block2') and TASK_ENV.block2 is not None:
        block2_pose = TASK_ENV.block2.get_pose()
        obs["block2_pose"] = block2_pose.p.tolist() + block2_pose.q.tolist()

    # Get action from policy
    action_result = model.predict_action(obs)
    action = action_result["action"]

    # Execute action
    TASK_ENV.take_action(action, action_type='qpos')


def reset_model(model):
    """
    Reset the model state if needed
    """
    model.reset()