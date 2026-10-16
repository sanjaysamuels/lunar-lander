"""Step 2: watch a random rocket crash and print the eight numbers.

    python watch_random.py            # 3 episodes with the window on
    python watch_random.py --episodes 1 --no-render
"""

import argparse

import gymnasium as gym

NAMES = [
    "x (left/right of pad)",
    "y (height above pad)",
    "sideways speed",
    "vertical speed",
    "tilt (radians)",
    "spin rate",
    "left leg down",
    "right leg down",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--print-every", type=int, default=25, help="frames between printouts")
    args = p.parse_args()

    env = gym.make("LunarLander-v3", render_mode=None if args.no_render else "human")
    for ep in range(args.episodes):
        obs, info = env.reset()
        total, frame = 0.0, 0
        while True:
            action = env.action_space.sample()  # 0 nothing, 1 left, 2 main, 3 right
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if frame % args.print_every == 0:
                print(f"\nframe {frame}  action={action}  reward={reward:+.2f}")
                for name, value in zip(NAMES, obs):
                    print(f"  {name:<22} {value:+.2f}")
            frame += 1
            if terminated or truncated:
                why = "crashed/landed" if terminated else "timed out"
                print(f"\nepisode {ep + 1}: {frame} frames, score {total:.1f} ({why})")
                break
    env.close()


if __name__ == "__main__":
    main()
