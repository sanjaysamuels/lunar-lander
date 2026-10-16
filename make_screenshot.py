"""Render a landing from a trained model to assets/landing.png and assets/landing.gif.

    python make_screenshot.py --name none
"""

import argparse

from PIL import Image
from stable_baselines3 import PPO

from rewards import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="none")
    p.add_argument("--seed", type=int, default=101)
    args = p.parse_args()

    model = PPO.load(f"models/{args.name}")
    env = make_env("none", render_mode="rgb_array")
    obs, info = env.reset(seed=args.seed)
    frames, total = [], 0.0
    while True:
        frames.append(Image.fromarray(env.render()))
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            frames.append(Image.fromarray(env.render()))
            break
    env.close()

    # Screenshot: the moment of touchdown.
    frames[-1].save("assets/landing.png")

    # GIF: every 3rd frame, half size, ~17 fps so it plays near real speed.
    small = [f.resize((f.width // 2, f.height // 2)) for f in frames[::3]]
    small[0].save("assets/landing.gif", save_all=True, append_images=small[1:],
                  duration=60, loop=0, optimize=True)
    print(f"score {total:.1f}, {len(frames)} frames -> assets/landing.png, assets/landing.gif")


if __name__ == "__main__":
    main()
