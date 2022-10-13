from stable_baselines3 import TD3


class TD3RL2:

    def __init__(self, **model_kwargs):
        print("Seed as seen by TD3: ", self.seed)
        self.model = TD3('MlpPolicy', self.env, seed=self.seed, **model_kwargs)
        self.model.set_random_seed(seed=self.seed)
        self.is_trained = False
        self.is_solved = None


    def learn(self, learning_timesteps=int(1e5)):
        self.model.learn(total_timesteps=learning_timesteps, log_interval=10,
                         tb_log_name='TD3')
        self.is_trained = True
        return {'total_steps': self.env.get_total_steps(),
                'episode_lengths': self.env.get_episode_lengths(),
                'episode_rewards': self.env.get_episode_rewards()}

    def select_eval_action(self, obs, **predict_kwargs):
        action = self.model.predict(obs, deterministic=True, **predict_kwargs)
        return action

    def save(self, filename):
        self.model.save(filename)
        print('TD3 saved to ' + filename)

    def load(self, filename, device='auto'):
        del self.model
        self.model = TD3.load(filename, device=device)
        print('TD3 loaded from ' + filename)
        self.is_trained = True

    def set_algo_seed(self, seed=None):
        self.model.set_random_seed(seed)
