import gymnasium as gym
import numpy as np


def make_env(env_id, idx, run_name, gamma=0.99, xml="ant.xml"):
    def thunk():
        if idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", xml_file=xml)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}+{xml}", episode_trigger=lambda x: x % 1000 == 0,
                                           disable_logger=True)
        else:
            env = gym.make(env_id, xml_file=xml)

        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), observation_space=None)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


if __name__ == '__main__':
    env = gym.vector.SyncVectorEnv([make_env("Ant-v5", 5+i, "a") for i in range(6)])
    obs, _ = env.reset()
    obs = np.delete(obs, (3,5,6), axis=1)
    print(obs.shape)


