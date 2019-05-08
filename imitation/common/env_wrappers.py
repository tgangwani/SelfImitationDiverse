import gym

class EpisodicEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        print('== Episodic Wrapper ==')
        gym.Wrapper.__init__(self, env)
        self.total_episode_reward = 0.

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.total_episode_reward += reward

        if done:
            reward = self.total_episode_reward
            self.total_episode_reward = 0.  # reset
        else:
            reward = 0.

        return observation, reward, done, info

class TimeLimitMaskWrapper(gym.Wrapper):
    """
    Checks whether done was caused my timit limits or not
    """
    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:  # pylint: disable=protected-access
            info['timestep_limit_reached'] = True

        return observation, reward, done, info
