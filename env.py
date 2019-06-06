
# python 3

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class Market_environmet(gym.Env):

    def __init__(self, entire_price_features, n_outputs, init_cash, stock_price_mean, stock_price_std):

        """
            Arguments
            - entire_price_features : entire sequence of price features
                                    [close/close_previous, cumulative_ratio]
            - n_outpus : number of output dimensions ( = number of actions)
            - init_cash : initial cash in hand
            - stock_price_mean : value of train stock price data
            - stock_price_std : deviation of train stock price data
        """

        self.entire_price_features = entire_price_features
        self.init_cash = init_cash

        self.train_mean = stock_price_mean
        self.train_std = stock_price_std

        # action space 정의
        self.action_space = spaces.Discrete(n_outputs)

        # observation space 정의 (the structure of the observation)
        self.observation_space = spaces.Box(low=np.array([0,0,0]), high = np.array([50.0,100000,10000]))

        # seed and initialize
        self._seed()
        self._reset()

    def _seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)

    def _reset(self):

        """
            Initialize the environment state
        """

        self.current_step = 0
        self.cash_in_hand = self.init_cash
        self.price_features = self.entire_price_features[0]
        self.n_stocks = 0
        return self._get_obs()

    def _get_obs(self):

        """
            Get the observations
        """
        obs = []
        obs.append(self.n_stocks)
        obs.append(self.cash_in_hand)
        obs.extend(self.price_features)
        return np.array(obs)

    def _step(self, action):

        """
            Change the environment state based on given action
        """

        assert self.action_space.contains(action)

        reward = self._calculate_reward(action)
        done = self.current_step == (len(self.entire_price_features) - 1)
        info = {'current_value':self._get_estimated_value()}
        obs = self._get_obs()

        return obs, reward, done, info

    def _get_estimated_value(self):
        return self.cash_in_hand + self.n_stocks * self._get_stock_price()

    def _get_stock_price(self):
        return self.price_features[1] * self.train_std + self.train_mean

    def _calculate_reward(self, action):

        """
            action
            - buy : 0 / sell : 1 / hold : 2
            price features
            - [close/previous_close, cumulative_ratio, sma]
        """

        current_stock_price = self._get_stock_price()

        self.current_step += 1
        self.price_features = self.entire_price_features[self.current_step]

        next_stock_price = self._get_stock_price()

        if action == 0:
            if self.cash_in_hand > current_stock_price:
                number_of_buyable = np.floor(self.cash_in_hand / current_stock_price)
                reward = (next_stock_price - current_stock_price) * number_of_buyable
                self.n_stocks = self.n_stocks + number_of_buyable
                self.cash_in_hand = self.cash_in_hand - (number_of_buyable * current_stock_price)
            else:
                reward = 0
        elif action == 1:
            if self.n_stocks > 0:
                reward = (current_stock_price - next_stock_price) * self.n_stocks
                self.cash_in_hand = self.cash_in_hand + self.n_stocks * current_stock_price
                self.n_stocks = 0
            else:
                reward = 0
        else:
            number_of_buyable = np.floor(self.cash_in_hand / current_stock_price)
            reward = self.n_stocks * (next_stock_price - current_stock_price) \
                     + number_of_buyable * (current_stock_price - next_stock_price)
        return reward