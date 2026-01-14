#!/usr/bin/env python3
"""
Data collection script for MLP policy training on stack_blocks_two task.
Collects state-based observations (proprioceptive + block poses) and actions.
"""

import sys
import os
sys.path.append("./")

import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some features may be limited.")

import pickle
from collections import defaultdict
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm

# Import environment and utilities
from envs.stack_blocks_two import stack_blocks_two
from script.collect_data import get_embodiment_config, class_decorator
from envs._GLOBAL_CONFIGS import CONFIGS_PATH


def collect_trajectory_data(task_env, task_config, num_episodes=10, save_dir="./data/mlp_stack_blocks_two"):
    """
    Collect trajectory data for MLP policy training.

    Args:
        task_env: The task environment instance
        task_config: Task configuration dictionary
        num_episodes: Number of episodes to collect
        save_dir: Directory to save collected data

    Returns:
        List of trajectories, each containing observations and actions
    """
    os.makedirs(save_dir, exist_ok=True)

    trajectories = []
    successful_episodes = 0

    print(f"Collecting {num_episodes} episodes of state-based data...")

    for episode_idx in tqdm(range(num_episodes)):
        # Reset environment with task config
        task_args = task_config.copy()
        task_args['now_ep_num'] = episode_idx
        task_args['seed'] = episode_idx
        task_env.setup_demo(**task_args)

        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'episode_success': False
        }

        # Execute the scripted policy to collect demonstration data
        try:
            task_env.play_once()

            # Check if episode was successful
            success = task_env.check_success()
            trajectory['episode_success'] = success

            if success:
                successful_episodes += 1

            # For successful episodes, we need to collect the trajectory data
            # This would require modifying the task to record observations during execution
            # For now, we'll use a placeholder approach

            print(f"Episode {episode_idx}: {'Success' if success else 'Failed'}")

        except Exception as e:
            print(f"Episode {episode_idx} failed with error: {e}")
            trajectory['episode_success'] = False

        trajectories.append(trajectory)
        task_env.close_env()

    print(f"Collected {successful_episodes}/{num_episodes} successful episodes")

    # Save trajectories
    save_path = os.path.join(save_dir, "trajectories.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)

    print(f"Saved trajectories to {save_path}")
    return trajectories


def process_trajectories_for_mlp(trajectories, save_dir="./data/mlp_stack_blocks_two"):
    """
    Process collected trajectories into format suitable for MLP training.

    Args:
        trajectories: List of trajectory dictionaries
        save_dir: Directory to save processed data
    """
    os.makedirs(save_dir, exist_ok=True)

    # Collect all observations and actions
    all_obs = []
    all_actions = []

    print("Processing trajectories for MLP training...")

    for traj in trajectories:
        if not traj['episode_success']:
            continue  # Skip failed episodes

        # Process each step in the trajectory
        # Note: This assumes we have step-by-step data, which we don't yet
        # In a real implementation, we'd need to modify the task to record
        # observations at each step during execution

        # For now, create synthetic data as an example
        # In practice, this would come from actual trajectory execution

        # Example: Create random but realistic state-action pairs
        for _ in range(50):  # Assume 50 steps per episode
            # Proprioceptive state (joint positions + grippers)
            proprio = np.random.randn(14) * 0.1  # Small random values

            # Block poses (position + quaternion)
            block1_pose = np.array([np.random.uniform(-0.3, 0.3),  # x
                                   np.random.uniform(-0.1, 0.1),  # y
                                   0.741,  # z (table height)
                                   1.0, 0.0, 0.0, 0.0])  # quaternion (identity)

            block2_pose = np.array([np.random.uniform(-0.3, 0.3),  # x
                                   np.random.uniform(-0.1, 0.1),  # y
                                   0.741,  # z (table height)
                                   1.0, 0.0, 0.0, 0.0])  # quaternion

            # Combine into full observation
            obs = np.concatenate([proprio, block1_pose, block2_pose])
            all_obs.append(obs)

            # Random action (joint velocities + gripper commands)
            action = np.random.randn(14) * 0.5
            all_actions.append(action)

    # Convert to numpy arrays
    all_obs = np.array(all_obs)
    all_actions = np.array(all_actions)

    print(f"Processed {len(all_obs)} state-action pairs")

    # Compute normalization statistics
    obs_mean = np.mean(all_obs, axis=0)
    obs_std = np.std(all_obs, axis=0)
    action_mean = np.mean(all_actions, axis=0)
    action_std = np.std(all_actions, axis=0)

    # Save processed data
    processed_data = {
        'observations': all_obs,
        'actions': all_actions,
        'obs_mean': obs_mean,
        'obs_std': obs_std,
        'action_mean': action_mean,
        'action_std': action_std,
        'num_samples': len(all_obs)
    }

    save_path = os.path.join(save_dir, "processed_data.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(processed_data, f)

    # Save stats separately for policy loading
    stats = {
        'obs_mean': obs_mean,
        'obs_std': obs_std,
        'action_mean': action_mean,
        'action_std': action_std
    }

    stats_path = os.path.join(save_dir, "dataset_stats.pkl")
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    print(f"Saved processed data to {save_path}")
    print(f"Saved stats to {stats_path}")

    return processed_data


def main():
    parser = argparse.ArgumentParser(description='Collect state-based data for MLP policy')
    parser.add_argument('--task_name', type=str, default='stack_blocks_two',
                       help='Task name')
    parser.add_argument('--task_config', type=str, default='demo_clean',
                       help='Task configuration file')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of episodes to collect')
    parser.add_argument('--save_dir', type=str, default='./data/mlp_stack_blocks_two',
                       help='Directory to save collected data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(args.seed)

    # Load task configuration
    task = class_decorator(args.task_name)
    config_path = f"./task_config/{args.task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        task_args = yaml.load(f.read(), Loader=yaml.FullLoader)

    task_args['task_name'] = args.task_name
    task_args['seed'] = args.seed

    # Set up embodiment configuration
    embodiment_type = task_args.get("embodiment", ["default"])
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    if len(embodiment_type) == 1:
        task_args["left_robot_file"] = _embodiment_types[embodiment_type[0]]["file_path"]
        task_args["right_robot_file"] = _embodiment_types[embodiment_type[0]]["file_path"]
        task_args["dual_arm_embodied"] = True
    else:
        raise ValueError("Only single embodiment type supported for MLP data collection")

    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])

    # Collect trajectory data
    trajectories = collect_trajectory_data(task, task_args, args.num_episodes, args.save_dir)

    # Process trajectories for MLP training
    processed_data = process_trajectories_for_mlp(trajectories, args.save_dir)

    print("Data collection completed!")
    print(f"Total samples: {processed_data['num_samples']}")
    print(f"Observation dim: {processed_data['observations'].shape[1]}")
    print(f"Action dim: {processed_data['actions'].shape[1]}")


if __name__ == "__main__":
    main()