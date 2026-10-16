"""Step 4: record a trained model flying, to videos/<name>-episode-N.mp4.

    python record.py --name none
    python record.py --name hover --episodes 2
"""

import argparse

import gymnasium as gym
from stable_baselines3 import PPO

from rewards import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="none")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args()

    model = PPO.load(f"models/{args.name}")
    env = gym.wrappers.RecordVideo(
        make_env("none", render_mode="rgb_array"),
        video_folder="videos",
        name_prefix=args.name,
        episode_trigger=lambda ep: True,
    )
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        print(f"episode {ep}: score {total:.1f} -> videos/{args.name}-episode-{ep}.mp4")
    env.close()


if __name__ == "__main__":
    main()
