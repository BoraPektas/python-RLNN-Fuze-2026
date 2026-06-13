# train.py
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import our custom missile environment
from environment import MissileEnv

import json
import time
import torch
from stable_baselines3.common.callbacks import BaseCallback

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total = total_timesteps
        self.start_time = time.time()
        
    def _on_step(self):
        if self.num_timesteps % 1000 == 0:
            rew = 0.0
            if len(self.model.ep_info_buffer) > 0:
                rew = sum(ep_info["r"] for ep_info in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
            elapsed = time.time() - self.start_time
            fps = int(self.num_timesteps / max(0.1, elapsed))
            
            data = {"step": self.num_timesteps, "total": self.total, "rew": rew, "fps": fps}
            try:
                import os
                path = os.path.join(os.path.dirname(__file__), "..", "train_progress.json")
                with open(path, "w") as f:
                    json.dump(data, f)
            except Exception:
                pass
        return True

def make_env(rank):
    def _init():
        return Monitor(MissileEnv(render_mode=None))
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-test", action="store_true", help="Skip the post-training interactive test run")
    parser.add_argument("--steps", type=int, default=5000000, help="Number of training timesteps")
    parser.add_argument("save_path", nargs="?", default="missile_ppo_model", help="Path to save the trained model")
    args = parser.parse_args()

    # 1. Initialize the Custom Missile Environment
    # render_mode=None: Optimized for training (no GUI overhead)
    # Parallelization: We spin up 16 parallel CPU environments to massively increase data throughput to the GPU!
    num_envs = 16
    try:
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)], start_method='spawn')
    except RuntimeError:
        # Fallback if spawn fails on some systems
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # 2. Configure the PyTorch-based PPO Algorithm
    # "MlpPolicy" automatically creates a standard feedforward neural network.
    # PPO is highly effective for continuous action spaces like missile steering.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=512,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        target_kl=0.02,
        n_epochs=10,
        device="cuda",
        policy_kwargs={"net_arch": [dict(pi=[256, 256], vf=[256, 256])]},
    )


    if torch.cuda.is_available():
        print(f"Hardware: CUDA in use (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print("Hardware: CPU in use")

    # 3. Start Training
    total_steps = args.steps
    print(f"Missile AI Training Starting... ({total_steps} timesteps)")
    model.learn(total_timesteps=total_steps, progress_bar=True, callback=ProgressCallback(total_steps))
    print("Training finished!")

    # Save model for GUI usage
    model.save(args.save_path)
    print(f"Model saved as '{args.save_path}'")

    env.close()


if __name__ == "__main__":
    main()