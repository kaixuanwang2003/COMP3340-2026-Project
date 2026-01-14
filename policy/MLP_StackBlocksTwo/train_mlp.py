#!/usr/bin/env python3
"""
Training script for MLP policy on stack_blocks_two task.
"""

import sys
import os
sys.path.append("./")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import yaml


class StateActionDataset(Dataset):
    """Dataset for state-action pairs"""

    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.observations = torch.tensor(data['observations'], dtype=torch.float32)
        self.actions = torch.tensor(data['actions'], dtype=torch.float32)

        # Normalization stats
        self.obs_mean = torch.tensor(data['obs_mean'], dtype=torch.float32)
        self.obs_std = torch.tensor(data['obs_std'], dtype=torch.float32)
        self.action_mean = torch.tensor(data['action_mean'], dtype=torch.float32)
        self.action_std = torch.tensor(data['action_std'], dtype=torch.float32)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = (self.observations[idx] - self.obs_mean) / (self.obs_std + 1e-8)
        action = (self.actions[idx] - self.action_mean) / (self.action_std + 1e-8)
        return obs, action


def train_mlp_policy(data_path, config, save_dir="./policy/MLP_StackBlocksTwo/checkpoints"):
    """
    Train the MLP policy.

    Args:
        data_path: Path to processed data file
        config: Training configuration
        save_dir: Directory to save checkpoints
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    dataset = StateActionDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Create model
    from .mlp_policy import StateMLP
    model = StateMLP(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        hidden_dims=config['hidden_dims']
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()

    # Device
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"Training MLP policy on {len(dataset)} samples")
    print(f"Model: {model}")

    # Training loop
    model.train()
    for epoch in range(config['num_epochs']):
        epoch_loss = 0.0
        num_batches = 0

        for obs, target_action in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            obs, target_action = obs.to(device), target_action.to(device)

            # Forward pass
            pred_action = model(obs)

            # Compute loss
            loss = criterion(pred_action, target_action)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"policy_epoch_{epoch+1}.ckpt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_checkpoint_path = os.path.join(save_dir, "policy_last.ckpt")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Saved final model: {final_checkpoint_path}")

    # Save dataset stats for policy loading
    stats = {
        'obs_mean': dataset.obs_mean.numpy(),
        'obs_std': dataset.obs_std.numpy(),
        'action_mean': dataset.action_mean.numpy(),
        'action_std': dataset.action_std.numpy()
    }

    stats_path = os.path.join(save_dir, "dataset_stats.pkl")
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"Saved dataset stats: {stats_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train MLP policy for stack_blocks_two')
    parser.add_argument('--data_path', type=str, default='./data/mlp_stack_blocks_two/processed_data.pkl',
                       help='Path to processed data file')
    parser.add_argument('--config', type=str, default='./policy/MLP_StackBlocksTwo/deploy_policy.yml',
                       help='Training configuration file')
    parser.add_argument('--save_dir', type=str, default='./policy/MLP_StackBlocksTwo/checkpoints/stack_blocks_two',
                       help='Directory to save checkpoints')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Train the policy
    trained_model = train_mlp_policy(args.data_path, config, args.save_dir)

    print("Training completed!")


if __name__ == "__main__":
    main()