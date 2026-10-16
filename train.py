"""Step 3 and 5: train PPO on LunarLander-v3, optionally through a reward wrapper.

    python train.py                              # baseline, 1M steps
    python train.py --reward time --name time    # with a per-frame fee
    python train.py --reward hover --name hover  # the broken rule
    python train.py --reward fixed --name fixed  # the fix

Prints ep_rew_mean (average shaped score over the last 100 episodes) and
rollout/raw_score_mean (the true game score, wrapper ignored). Saves to
models/<name>.zip and a CSV of the curve to logs/<name>/progress.csv.
"""

import argparse
import time
from collections import deque

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from rewards import REWARDS, make_env


class RawScoreCallback(BaseCallback):
    """Logs the unshaped game score per episode so hacking is visible."""

    def __init__(self):
        super().__init__()
        self.running = None
        self.scores = deque(maxlen=100)

    def _on_training_start(self):
        self.running = np.zeros(self.training_env.num_envs)

    def _on_step(self):
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        for i, info in enumerate(infos):
            self.running[i] += info.get("raw_reward", self.locals["rewards"][i])
            if dones[i]:
                self.scores.append(self.running[i])
                self.running[i] = 0.0
        return True

    def _on_rollout_end(self):
        if self.scores:
            self.logger.record("rollout/raw_score_mean", float(np.mean(self.scores)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reward", choices=REWARDS, default="none")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--name", default=None, help="model/log name (default: reward name)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--envs", type=int, default=16, help="parallel environments")
    args = p.parse_args()
    name = args.name or args.reward

    # Wrapper goes inside, so Monitor (and ep_rew_mean) sees the shaped reward PPO learns from.
    venv = make_vec_env(lambda: make_env(args.reward), n_envs=args.envs, seed=args.seed)

    # Hyperparameters from the SB3 RL Zoo tuned for LunarLander.
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        seed=args.seed,
        verbose=1,
    )
    model.set_logger(configure(f"logs/{name}", ["stdout", "csv"]))

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=RawScoreCallback())
    minutes = (time.time() - t0) / 60

    model.save(f"models/{name}")
    print(f"\nsaved models/{name}.zip after {args.steps:,} steps in {minutes:.1f} min")
    print(f"next: python evaluate.py --name {name}")


if __name__ == "__main__":
    main()
