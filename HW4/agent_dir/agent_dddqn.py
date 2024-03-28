from agent_dir.agent import Agent
from models.dddqn import DDDQN
from common.utils import init_weights
from common.replay_buffer import ReplayBuffer

import os
import random
import numpy as np
from collections import deque

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter


class Agent_DDDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DDDQN, self).__init__(env)

        ##################
        # YOUR CODE HERE #
        ##################
        INPUT_SHAPE = (4, 84, 84)
        OUTPUT_DIM = env.action_space.n
        HID_DIM = 512

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DDDQN(INPUT_SHAPE, HID_DIM, OUTPUT_DIM).to(self.device)

        if args.test_dddqn:
            # you can load your model here
            print('Loading trained model...')
            self._load_model()
            print(self.model)
            return

        self.env = env
        self.args = args
        self.logger = SummaryWriter('./log/4-2/dddqn')
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.replay_buffer = ReplayBuffer(10000)
        self.target_model = DDDQN(INPUT_SHAPE, HID_DIM, OUTPUT_DIM).to(self.device)

        self.model.apply(init_weights)
        self._update_target(self.model, self.target_model)
        
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
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        num_steps = self.args.num_steps
        batch_size = self.args.batch_size
        train_freq = self.args.train_freq
        update_freq = self.args.update_freq
        
        step = 1
        episode = 0
        episode_reward = 0
        early_stop = False
        STOP_SCORE = 30
        NUM_TRIALS = 30
        train_rewards = deque(maxlen=NUM_TRIALS)

        state = self.env.reset()
        while step < num_steps+1 and not early_stop:
            self.epsilon = self._epsilon_by_step(step)
            action = self.make_action(state, test=False)
            next_state, reward, done, _ = self.env.step(action)
            
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if self.replay_buffer.is_full() and step % train_freq == 0:
                self._update_model(batch_size)

            if step % update_freq == 0:
                self._update_target(self.model, self.target_model)

            if done:
                # log
                episode += 1
                train_rewards.append(episode_reward)
                mean_train_reward = np.mean(train_rewards)
                log = {'episode reward': episode_reward,
                       'mean episode reward': mean_train_reward}
                self.logger.add_scalars('DDDQN', log, episode)
                self.logger.add_scalar('mean episode reward', mean_train_reward, episode)

                if mean_train_reward >= STOP_SCORE: early_stop = True
                
                episode_reward = 0
                state = self.env.reset()

                # save
                if episode % 10 == 0 or early_stop:
                    self._save_model()
                    
                if episode % 100 == 0 or early_stop:
                    print(f'Episode: {episode:6,}, Step: {step:9,} | Reward: {log["episode reward"]:5.1f} | Mean Reward: {mean_train_reward:5.1f}')
                    
            if step == num_steps:
                print(f'Episode: {episode:6,}, Step: {step:9,} | Reward: {log["episode reward"]:5.1f} | Mean Reward: {mean_train_reward:5.1f}')

            step += 1


    @torch.no_grad()
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        observation = self._preprocess_state(observation)
        q_pred = self.model(observation)

        epsilon = 0.01 if test else self.epsilon

        if random.random() > epsilon:
            action = q_pred.argmax(dim=1).item()
        else:
            action = self.env.action_space.sample()

        return action


    def _preprocess_state(self, state):
        """
        Turn shape from (84, 84, 4) to (1, 4, 84, 84).
        """
        return torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, 4, 84, 84)


    def _save_model(self, save_path='./saved_models/dddqn'):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, 'model_dddqn.pt'))
        # print(f'Save model to {save_path}')


    def _load_model(self, load_path='./saved_models/dddqn'):
        self.model.load_state_dict(torch.load(os.path.join(load_path, 'model_dddqn.pt'), map_location=self.device))
        # print(f'Load model from {load_path}')


    def _epsilon_by_step(self, step):
        """
        2-phase linear schedule of epsilon:
        step       1 ~  25_000: linear decline from 1 to 0.1
        step  25_001 ~ 500_000: linear decline from 0.1 to 0.01
        step 500_001 afterward: 0.01
        """
        if step <= 25_000:
            epsilon_start = 1
            epsilon_end = 0.1
            eps_steps = 25000
        else:
            epsilon_start = .1
            epsilon_end = 0.01
            eps_steps = 500000
        slope = (epsilon_end - epsilon_start) / eps_steps
        return max(epsilon_end, epsilon_start + step * slope)


    def _update_target(self, model, target_model):
        target_model.load_state_dict(model.state_dict())


    def _update_model(self, batch_size, gamma=0.99):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_preds = self.model(states)
        next_q_preds = self.model(next_states)
        next_actions = next_q_preds.argmax(dim=1)
        next_q_preds_target = self.target_model(next_states)

        # use Q function from the Double DQN paper:
        # using the original model to select action, and using the target model to evaluate the selected action.
        q_values = q_preds.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
        next_q_values = next_q_preds_target.gather(dim=1, index=next_actions.unsqueeze(1)).squeeze(1).detach()
        expected_q_values = rewards + gamma * next_q_values * (1 - dones)

        loss = (q_values - expected_q_values).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()