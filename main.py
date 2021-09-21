import gym
import json
import datetime as dt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from env.IRSwap_10Y import IRSwap_10Y
import pandas as pd


df = pd.read_csv('./data/IRS_10.csv')

env = DummyVecEnv([lambda: IRSwap_10Y(df)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=2000)

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
