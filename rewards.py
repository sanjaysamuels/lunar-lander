"""Reward wrappers for LunarLander-v3.

A wrapper sits between the game and PPO. Each step it takes the built-in
reward, adds or subtracts its own terms, and passes the total on. The
observation is the same eight numbers the article describes:

    obs[0] x position      obs[4] tilt (radians)
    obs[1] y position      obs[5] spin rate
    obs[2] sideways speed  obs[6] left leg on ground (0/1)
    obs[3] vertical speed  obs[7] right leg on ground (0/1)

Every wrapper here stores the untouched game reward in info["raw_reward"]
so training logs can show the true score next to the shaped one.
"""

import math

import gymnasium as gym


class RewardShaper(gym.Wrapper):
    """Base class. Subclasses override `bonus(obs, reward, terminated)`."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.on_reset(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["raw_reward"] = float(reward)
        shaped = reward + self.bonus(obs, reward, terminated)
        return obs, shaped, terminated, truncated, info

    def on_reset(self, obs):
        pass

    def bonus(self, obs, reward, terminated):
        return 0.0


class TimePenalty(RewardShaper):
    """Step 5 of the article: a tiny charge every frame for taking too long.

    The built-in reward has no clock, so a rocket can dawdle for free.
    """

    def __init__(self, env, fee=0.05):
        super().__init__(env)
        self.fee = fee

    def bonus(self, obs, reward, terminated):
        return -self.fee


def _upright_and_slow(obs, max_tilt=0.2, max_speed=0.3):
    tilt = abs(float(obs[4]))
    speed = math.hypot(float(obs[2]), float(obs[3]))
    return tilt < max_tilt and speed < max_speed


class HoverBonus(RewardShaper):
    """Step 6, the broken rule: +1 for every frame the rocket is upright and slow.

    An episode is cut off at 1000 frames, so hanging in the air can collect up
    to 1000 points from this rule and pay at most about 300 in fuel. Landing
    ends the episode and the payments stop, so hovering can beat landing.
    This is reward hacking, on purpose. Watch the clip, not just the score.
    """

    def __init__(self, env, bonus=1.0):
        super().__init__(env)
        self.per_frame = bonus

    def bonus(self, obs, reward, terminated):
        return self.per_frame if _upright_and_slow(obs) else 0.0


class HoverFixed(RewardShaper):
    """Step 6, the fix: pay for getting better, not for sitting in a nice spot.

    Niceness is a single number (negative tilt plus speed, so bigger is
    better). The bonus each frame is how much niceness *improved* since the
    last frame, times a scale. Sitting still improves nothing, so it pays
    nothing. A small fee every frame makes waiting cost something.
    """

    def __init__(self, env, scale=10.0, fee=0.05):
        super().__init__(env)
        self.scale = scale
        self.fee = fee
        self.prev = None

    @staticmethod
    def niceness(obs):
        tilt = abs(float(obs[4]))
        speed = math.hypot(float(obs[2]), float(obs[3]))
        return -(tilt + speed)

    def on_reset(self, obs):
        self.prev = self.niceness(obs)

    def bonus(self, obs, reward, terminated):
        now = self.niceness(obs)
        delta = now - self.prev
        self.prev = now
        return self.scale * delta - self.fee


# Names used by train.py --reward
REWARDS = {
    "none": None,
    "time": TimePenalty,
    "hover": HoverBonus,
    "fixed": HoverFixed,
}


def make_env(reward="none", render_mode=None):
    """LunarLander-v3 with the chosen wrapper applied (or none)."""
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    wrapper = REWARDS[reward]
    return wrapper(env) if wrapper else env
