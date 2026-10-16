"""How you know it works: average true score over many attempts, wrapper OFF.

    python evaluate.py --name none              # 20 episodes, no window
    python evaluate.py --name hover --render    # watch it (does it hover?)

200 or more on average counts as solved.
"""

import argparse

import numpy as np
from stable_baselines3 import PPO

from rewards import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="none")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--render", action="store_true")
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args()

    model = PPO.load(f"models/{args.name}")
    env = make_env("none", render_mode="human" if args.render else None)

    scores, lengths, landed = [], [], 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total, frames = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            frames += 1
            if terminated or truncated:
                break
        ok = terminated and reward > 0  # ended with the +100 resting bonus
        landed += ok
        scores.append(total)
        lengths.append(frames)
        tag = "landed" if ok else ("crashed" if terminated else "TIMED OUT")
        print(f"episode {ep + 1:>3}: score {total:7.1f}  frames {frames:4d}  {tag}")
    env.close()

    print(f"\n{args.name}: mean {np.mean(scores):.1f} +/- {np.std(scores):.1f} "
          f"over {args.episodes} episodes, landed {landed}/{args.episodes}, "
          f"mean length {np.mean(lengths):.0f} frames")
    print("solved" if np.mean(scores) >= 200 else "not solved yet (need mean >= 200)")


if __name__ == "__main__":
    main()
