import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import copy
Max_delta = 10000
Max_reward = 10000000000
Max_value = 10
B1_limit = 10000
B2_limit = 10000
B3_limit = 15000
B4_limit = 20000
B_total = 25000
time_max = 1000

Max_steps = 20000

class IRSwap_10Y(gym.Env):
    """An IRS SWAP Trading for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        """ Initialisation of all variables of our Model"""
        super(IRSwap_10Y, self).__init__()
        self.df = df
        self.reward_range = (-Max_reward, Max_reward)
        self.action_space = spaces.Box(
            low=np.zeros(20), high=np.ones(20), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(6, 10), dtype=np.float16)
        open('output.csv','w')

    def _next_observation(self):
        """ Get the daliy rate of the Interst rate Swap
        for the last 3 days and scale to between 0-1"""
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        2, '1Y'].values / Max_value,
            self.df.loc[self.current_step: self.current_step +
                        2, '2Y'].values / Max_value,
            self.df.loc[self.current_step: self.current_step +
                        2, '3Y'].values / Max_value,
            self.df.loc[self.current_step: self.current_step +
                        2, '4Y'].values / Max_value,
            self.df.loc[self.current_step: self.current_step +
                        2, '5Y'].values / Max_value,
            self.df.loc[self.current_step: self.current_step +
                        2, '6Y'].values / Max_value,
            self.df.loc[self.current_step: self.current_step +
                        2, '7Y'].values / Max_value,
            self.df.loc[self.current_step: self.current_step +
                        2, '8Y'].values / Max_value,
            self.df.loc[self.current_step: self.current_step +
                        2, '9Y'].values / Max_value,
            self.df.loc[self.current_step: self.current_step +
                        2, '10Y'].values / Max_value,
        ]).T

        """ Append additional data and scale each value to between 0-1"""
        obs1 = np.append(frame,  np.reshape(self.delta, (1, 10)) / Max_delta , axis=0)
        obs2 = np.append(obs1, np.reshape(self.delta_variation, (1, 10)) / Max_delta , axis=0)
        obs = np.append( obs2, self.mtm * np.ones((1, 10))/ (Max_delta * Max_value), axis=0)
        return obs

    def _take_action(self, action):
        """ There is three actio for each maturity.
        Increase delta, decrease delta or left it constanat"""
        for i in range(10):
            action_type = action[2 * i]
            amount = action[2 * i + 1]
            if action_type < 1/3.0:
                self.delta_variation[i] = amount * abs(self.delta[i])
            elif action_type < 2/3.0:
                self.delta_variation[i] = -amount * abs(self.delta[i])
            else:
                self.delta_variation[i] = 0
        """ We add here a wite noise because we remark that the
        model give us quickly constant and null deltas """
        noise = (1/100) * np.random.randn(10) * B_total
        self.delta_variation += noise * self.delta
        self.delta += self.delta_variation

        
    def step(self, action):
        """ Execute one time step within the environment"""
        
        self._take_action(action)
        time = 0
        """ Before takin actions, we have to verify that our buckets are in the imposed ranges.
        Otherwise we have to take another action."""
        while ((abs(self.delta[0] +  self.delta[1]) > B1_limit)
               or (abs(self.delta[2] +  self.delta[3] +  self.delta[4]) > B2_limit)
               or (abs(self.delta[5] +  self.delta[6]) > B3_limit)
               or (abs(self.delta[7] +  self.delta[8] +  self.delta[9]) > B4_limit)
               or (abs(np.sum(self.delta)) > B_total)) and (time < time_max):
            self.delta -= self.delta_variation
            self._take_action(action)
            time += 1
        if time == time_max:
            self.delta -= self.delta_variation
            self.delta_variation = np.zeros(10)
        self.current_step += 1
        if self.current_step > len(self.df.loc[:, '1Y'].values) - 3:
            self.current_step = 0
        delay_modifier = self.current_step / Max_steps
        self.pnl += np.vdot(self.delta_variation, np.arange(1, 11)) * 0.05 * 0.01
        if self.current_step == 0:
            self.mtm = np.vdot(self.delta, np.array([self.df.loc[self.current_step, '1Y':].values ]).reshape(10))
        else:
            self.mtm = np.vdot(self.delta,
                            np.array([self.df.loc[self.current_step, '1Y':].values ]).reshape(10) - np.array([self.df.loc[self.current_step - 1, '1Y':].values ]).reshape(10)
                            )
        self.pnl += self.mtm
        """ Calculate reward, where reward increase if PnL increase """
        if self.pnl > self.pnl_ancien :
            reward = 100
        elif self.pnl == self.pnl_ancien:
            reward = -10
        else:
            reward = -100
        done = (self.pnl < -1000000)
        obs = self._next_observation()
        self.pnl_ancien = self.pnl
        return obs, reward, done, {}

    def reset(self):
        #Set delta
        self.delta = 0.01 * np.random.randn(10) * B1_limit
        # Set PNL
        self.pnl = 0
        self.step_numbers = 0
        self.pnl_ancien = 0
        self.mtm = 0
        self.delta_variation = np.zeros(10)
        self.current_step = random.randint(
            0, len(self.df.loc[:, '1Y'].values) - 3)
        self.step_init = copy.copy(self.current_step)
        return self._next_observation()
    

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        self.step_numbers = -(self.step_init - self.current_step)
        print(f'Step: {self.step_numbers}')
        print(f'PNL: {self.pnl}')
        print(f'deltas: {self.delta} ')
        sortie = str(self.current_step) + ','
        for i in range(10):
            sortie += str(self.delta[i]) + ','
        sortie += str(self.pnl) + '\n'
        open('output.csv','a').write(sortie)

            


