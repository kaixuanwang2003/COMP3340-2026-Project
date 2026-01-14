# MLP Policy for Stack Blocks Two Task

This is a state-based MLP (Multi-Layer Perceptron) policy implementation for the `stack_blocks_two` task in RoboTwin. The policy uses proprioceptive robot observations (joint states and gripper positions) combined with block pose information as input to predict robot actions.

## Overview

The MLP policy takes state-based observations and outputs joint position commands for bimanual robotic manipulation. Unlike vision-based policies, this approach focuses on learning from proprioceptive and object state information only.

### Observation Space
- **Proprioceptive states** (14 dimensions):
  - Left arm joints (6 dimensions)
  - Left gripper position (1 dimension)
  - Right arm joints (6 dimensions)
  - Right gripper position (1 dimension)
- **Block states** (14 dimensions):
  - Block 1 pose: position (3) + quaternion (4)
  - Block 2 pose: position (3) + quaternion (4)
- **Total observation dimension**: 28

### Action Space
- **Joint commands** (14 dimensions):
  - Left arm joint velocities (6 dimensions)
  - Left gripper command (1 dimension)
  - Right arm joint velocities (6 dimensions)
  - Right gripper command (1 dimension)

## Installation & Setup

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib tqdm pyyaml
```

### Data Collection
Collect state-based training data for the MLP policy:

```bash
# Collect demonstration trajectories
python policy/MLP_StackBlocksTwo/collect_state_data.py \
    --task_name stack_blocks_two \
    --task_config demo_clean \
    --num_episodes 100 \
    --save_dir ./data/mlp_stack_blocks_two

# Process trajectories for training
# (This is handled automatically in collect_state_data.py)
```

### Training
Train the MLP policy using the collected data:

```bash
python policy/MLP_StackBlocksTwo/train_mlp.py \
    --data_path ./data/mlp_stack_blocks_two/processed_data.pkl \
    --config ./policy/MLP_StackBlocksTwo/deploy_policy.yml \
    --save_dir ./policy/MLP_StackBlocksTwo/checkpoints/stack_blocks_two
```

## Usage

### Evaluation
Evaluate the trained MLP policy on the stack_blocks_two task:

```bash
# Run evaluation script
bash policy/MLP_StackBlocksTwo/eval.sh stack_blocks_two demo_clean default 42 0
```

### Configuration

The policy behavior can be configured via `deploy_policy.yml`:

```yaml
# Model configuration
obs_dim: 28          # Observation dimension
action_dim: 14       # Action dimension
hidden_dims: [256, 256, 256]  # MLP hidden layer dimensions
device: "cuda:0"     # Computing device
action_scale: 1.0    # Action scaling factor

# Training configuration
learning_rate: 1e-4
batch_size: 64
num_epochs: 100

# Data configuration
data_type:
  qpos: true         # Use joint positions
  endpose: false     # Don't use end effector poses
  rgb: false         # Don't use RGB images
  depth: false       # Don't use depth images
  pointcloud: false  # Don't use point clouds
```

## Architecture

The MLP policy consists of:

1. **StateMLP Model**: A feedforward neural network with configurable hidden layers
2. **Normalization**: Input/output normalization using dataset statistics
3. **State Observation Extraction**: Custom logic to extract relevant states from RoboTwin observations
4. **Action Scaling**: Safety scaling of predicted actions

### Model Details

```python
class StateMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256, 256]):
        # Input normalization layer
        # Hidden layers with ReLU activation and LayerNorm
        # Output layer for action prediction
```

## Key Features

- **State-Based**: Relies on proprioceptive and object state information only
- **Lightweight**: Simple MLP architecture suitable for real-time control
- **Configurable**: Easily adjustable network architecture and training parameters
- **Compatible**: Follows RoboTwin policy interface standards
- **Normalized**: Automatic input/output normalization for stable training

## Limitations

- Requires accurate state estimation of block poses
- May not generalize to tasks requiring visual reasoning
- Limited by the quality and quantity of collected demonstration data

## Troubleshooting

### Common Issues

1. **Missing block pose data**: Ensure the task environment provides block pose information in observations
2. **Normalization errors**: Check that dataset statistics are properly saved and loaded
3. **Action scaling**: Adjust `action_scale` parameter if actions are too aggressive

### Performance Tips

- Collect diverse demonstration data with domain randomization
- Experiment with different network architectures (hidden_dims)
- Monitor training loss and validation performance
- Use appropriate learning rates for stable convergence

## Citation

If you use this MLP policy implementation, please cite the RoboTwin paper:

```bibtex
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Lan, Zhiqian and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```