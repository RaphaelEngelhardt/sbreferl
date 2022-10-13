from stable_baselines3 import PPO


class PPORL2:

    def __init__(self, **model_kwargs):
        print("Seed as seen by PPO: ", self.seed)
        self.model = PPO('MlpPolicy', self.env, seed=self.seed, **model_kwargs)
        self.model.set_random_seed(seed=self.seed)
        self.is_trained = False
        self.is_solved = None


    def learn(self, learning_timesteps=int(1e6)):
        self.model.learn(total_timesteps=learning_timesteps, log_interval=10)
        self.is_trained = True
        return {'total_steps': self.env.get_total_steps(),
                'episode_lengths': self.env.get_episode_lengths(),
                'episode_rewards': self.env.get_episode_rewards()}

    def select_eval_action(self, obs, **predict_kwargs):
        action = self.model.predict(obs, deterministic=True, **predict_kwargs)
        return action

    def save(self, filename):
        self.model.save(filename)
        print('PPO saved to ' + filename)

    def load(self, filename, device='auto'):
        del self.model
        self.model = PPO.load(filename, device=device, seed=self.seed)
        print('PPO loaded from ' + filename)
        self.is_trained = True

    def set_algo_seed(self, seed=None):
        self.model.set_random_seed(seed)
