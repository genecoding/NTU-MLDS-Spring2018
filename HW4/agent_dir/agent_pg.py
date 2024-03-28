from agent_dir.agent import Agent
from models.mlp import MLP
from common.utils import init_weights

import os
import numpy as np
from collections import deque

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter


def prepro(I):
    """
    Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
    This preprocessing code is from
        https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

    Input:
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array
        (80 x 80) 1D float vector
    """
    
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float32).ravel()


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG, self).__init__(env)

        ##################
        # YOUR CODE HERE #
        ##################
        INPUT_DIM = 80 * 80
        OUTPUT_DIM = 2 #env.action_space.n
        HID_DIM = 256

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MLP(INPUT_DIM, HID_DIM, OUTPUT_DIM).to(self.device)

        if args.test_pg:
            # you can load your model here
            print('Loading trained model...')
            self._load_model()
            print(self.model)
            return

        self.env = env
        self.args = args
        self.logger = SummaryWriter('./log/4-1/pg')
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        self.model.apply(init_weights)
        print(self.model)
        print(self.args)


    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.prev_state = None


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        num_episodes = self.args.num_episodes
        
        episode = 1
        early_stop = False
        STOP_SCORE = 20
        NUM_TRIALS = 30
        train_rewards = deque(maxlen=NUM_TRIALS)

        while episode < num_episodes+1 and not early_stop:
            episode_reward = 0
            log_prob_actions = []
            rewards = []
            done = False
            self.prev_state = None
            state = self.env.reset()

            while not done:
                action, log_prob_action = self.make_action(state, test=False)
                state, reward, done, _ = self.env.step(action)

                episode_reward += reward
                log_prob_actions.append(log_prob_action)
                rewards.append(reward)

            returns = self._compute_returns(rewards)
            returns = torch.FloatTensor(returns).to(self.device)  # shape: (the length of a episode)
            returns = self._normalize(returns)
            log_prob_actions = torch.stack(log_prob_actions)  # (the length of a episode)
            self._update_model(log_prob_actions, returns)

            # log
            train_rewards.append(episode_reward)
            mean_train_reward = np.mean(train_rewards)  # compute mean of the last NUM_TRIALS rewards
            log = {'episode reward': episode_reward,
                   'mean episode reward': mean_train_reward}
            self.logger.add_scalars('PG', log, episode)
            self.logger.add_scalar('mean episode reward', mean_train_reward, episode)

            if mean_train_reward >= STOP_SCORE: early_stop = True

            # save
            if episode % 100 == 0 or early_stop:
                self._save_model()
                print(f'Episode: {episode:5} | Reward: {episode_reward:5.1f} | Mean Reward: {mean_train_reward:5.1f}')

            episode += 1


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        observation = self._preprocess_state(observation)
        dist = self.model(observation)
        sample = dist.sample()
        log_prob_action = dist.log_prob(sample)
        # if sample is 0, set action 2 (go up); else sample is 1, set action 3 (go down)
        action = 2 if sample.item() == 0 else 3

        if test:
            return action
        else:
            return action, log_prob_action


    def _preprocess_state(self, state):
        """
        Return a difference frame (subtraction of the current and the last frame)
        as input to the model.
        """
        state = prepro(state)
        x = state - self.prev_state if self.prev_state is not None else np.zeros_like(state)
        self.prev_state = state
        return torch.FloatTensor(x).to(self.device)  # (6400)


    def _save_model(self, save_path='./saved_models/pg'):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, 'model_pg.pt'))
        # print(f'Save model to {save_path}')


    def _load_model(self, load_path='./saved_models/pg'):
        self.model.load_state_dict(torch.load(os.path.join(load_path, 'model_pg.pt')))
        # print(f'Load model from {load_path}')


    def _compute_returns(self, rewards, gamma=0.99):
        returns = []
        R = 0

        for reward in reversed(rewards):
            R = reward + R * gamma
            returns.insert(0, R)

        return returns


    def _update_model(self, log_prob_actions, returns):
        loss = -(log_prob_actions * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def _normalize(self, data):
        return (data - data.mean()) / (data.std() + 1e-8)