import cv2
import numpy as np
import pandas as pd
from IPython.display import display

Arr = np.ndarray
cv2.ocl.setUseOpenCL(False)


def window_avg(arr: Arr, window: int):
    """
    Computes sliding window average
    """
    return np.convolve(arr, np.ones(window), mode="valid") / window


def cummean(arr: Arr):
    """
    Computes the cumulative mean
    """
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)


# Taken from https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
# See https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
def ewma(arr: Arr, alpha: float):
    """
    Returns the exponentially weighted moving average of x.
    Parameters:
    -----------
    x : array-like
    alpha : float {0 <= alpha <= 1}
    Returns:
    --------
    ewma: numpy array
          the exponentially weighted moving average
    """
    # Coerce x to an array
    s = np.zeros_like(arr)
    s[0] = arr[0]
    for i in range(1, len(arr)):
        s[i] = alpha * arr[i] + (1 - alpha) * s[i - 1]
    return s


def sum_rewards(rewards: list[int], gamma: float = 1):
    """
    Computes the total discounted sum of rewards for an episode.
    By default, assume no discount
    Input:
        rewards [r1, r2, r3, ...] The rewards obtained during an episode
        gamma: Discount factor
    Output:
        The sum of discounted rewards
        r1 + gamma*r2 + gamma^2 r3 + ...
    """
    total_reward = 0
    for r in rewards[:0:-1]:  # reverse, excluding first
        total_reward += r
        total_reward *= gamma
    total_reward += rewards[0]
    return total_reward


def arg_help(args, print_df=False):
    from part3_ppo.solutions import ARG_HELP_STRINGS, PPOArgs

    """Prints out a nicely displayed list of arguments, their default values, and what they mean."""
    if args is None:
        args = PPOArgs()
        changed_args = []
    else:
        default_args = PPOArgs()
        # print(default_args.__dict__)
        changed_args = [
            key for key in default_args.__dict__ if getattr(default_args, key) != getattr(args, key)
        ]
    df = pd.DataFrame([ARG_HELP_STRINGS]).T
    df.columns = ["description"]
    df["default value"] = [repr(getattr(args, name)) for name in df.index]
    df.index.name = "arg"
    df = df[["default value", "description"]]
    if print_df:
        df.insert(1, "changed?", ["yes" if i in changed_args else "" for i in df.index])
        with pd.option_context(
            "max_colwidth", 0, "display.width", 150, "display.colheader_justify", "left"
        ):
            print(df)
    else:
        s = df.style.set_table_styles(
            [
                {"selector": "td", "props": "text-align: left;"},
                {"selector": "th", "props": "text-align: left;"},
            ]
        ).apply(
            lambda row: ["background-color: red" if row.name in changed_args else None]
            + [
                None,
            ]
            * (len(row) - 1),
            axis=1,
        )
        with pd.option_context("max_colwidth", 0):
            display(s)


#NEP Aug 11 2025 I copied this from a deleted historical commit 
# https://github.com/callummcdougall/ARENA_3.0/commit/f25c7189c5f696fcebc1e267c88d1ea57d584118#diff-221a44b99bef5949898cd22e57fb557d01e032bd8e6f56e3030f99551bf4f74a
# Temporarily not using it since it's seem in the original repo they copied make_env from part2 isntead 
def make_env(
    env_id: str,
    seed: int,
    idx: int,
    run_name: str,
    mode: str = "classic-control",
    capture_video_every_n_episodes: int = None,
    video_save_path: str = None,
    **kwargs,
):
    """
    Return a function that returns an environment after setting up boilerplate.
    """

    # if capture_video_every_n_steps is None:
    #     video_log_freq = {"classic-control": 100, "atari": 30, "mujoco": 50}[mode]
    #     video_log_freq_step = {"classic-control": 2_000}[mode]

    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if idx == 0 and capture_video_every_n_episodes:
            env = gym.wrappers.RecordVideo(
                env,
                f"{video_save_path}/{run_name}",
                # use_wandb=use_wandb,
                episode_trigger=lambda episode_id: episode_id % capture_video_every_n_episodes == 0,
                disable_logger=True,
            )

        if mode == "atari":
            env = prepare_atari_env(env)
        elif mode == "mujoco":
            env = prepare_mujoco_env(env)

        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
