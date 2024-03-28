from agent_dir.agent import Agent
from models.actorcritic import ActorCritic
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


class Agent_PPO(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PPO, self).__init__(env)

        ##################
        # YOUR CODE HERE #
        ##################
        INPUT_DIM = (80 * 80) * 2
        OUTPUT_DIM = env.action_space.n
        HID_DIM = 256

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ActorCritic(INPUT_DIM, HID_DIM, OUTPUT_DIM).to(self.device)

        if args.test_ppo:
            # you can load your model here
            print('Loading trained model...')
            self._load_model()
            print(self.model)
            return

        self.env = env
        self.args = args
        self.logger = SummaryWriter('./log/4-1/ppo')
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
        mini_episodes = self.args.mini_episodes
        mini_steps = self.args.mini_steps
        ppo_epochs = self.args.ppo_epochs
        # mini_batch_size = self.args.mini_batch_size
        
        step = 0
        episode = 1
        early_stop = False
        STOP_SCORE = 20
        NUM_TRIALS = 30
        train_rewards = deque(maxlen=NUM_TRIALS)
        
        l_log_probs = []
        l_actions = []
        l_rewards = []
        l_returns = []
        l_values = []
        l_states = []
        l_masks = []

        while episode < num_episodes+1 and not early_stop:
            done = False
            episode_reward = 0
            self.prev_state = None
            state = self.env.reset()
            state = self._preprocess_state(state)

            while not done:
                action, log_prob, value = self.make_action(state, test=False)
                next_state, reward, done, _ = self.env.step(action.item())
                next_state = self._preprocess_state(next_state)

                episode_reward += reward
                l_log_probs.append(log_prob)
                l_actions.append(action)
                l_values.append(value)
                l_states.append(state)
                l_rewards.append(torch.FloatTensor([reward]).to(self.device))
                l_masks.append(torch.FloatTensor([1 - done]).to(self.device))
                
                state = next_state
                step += 1

                # calculate returns (with GAE) every 'mini_steps' steps
                if step % mini_steps == 0:
                    _, next_value = self.model(next_state)
                    l_returns += self._compute_returns(next_value, l_rewards[-mini_steps:], l_masks[-mini_steps:], l_values[-mini_steps:])

                    step = 0
                    
            # log
            train_rewards.append(episode_reward)
            mean_train_reward = np.mean(train_rewards)
            log = {'episode reward': episode_reward,
                   'mean episode reward': mean_train_reward}
            self.logger.add_scalars('PPO', log, episode)
            self.logger.add_scalar('mean episode reward', mean_train_reward, episode)
                
            if mean_train_reward >= STOP_SCORE: early_stop = True

            # update model every 'mini_episodes' episodes
            if episode % mini_episodes == 0:
                len_returns = len(l_returns)
    
                returns = torch.cat(l_returns).detach()  # (NUM_STEPS, 1)
                log_probs = torch.cat(l_log_probs[:len_returns]).detach()  # (NUM_STEPS)
                values = torch.cat(l_values[:len_returns]).detach()  # (NUM_STEPS, 1)
                actions = torch.cat(l_actions[:len_returns])  # (NUM_STEPS)
                states = torch.cat(l_states[:len_returns])  # (NUM_STEPS, 80*80*2)
                advantages = self._normalize(returns - values)  # (NUM_STEPS, 1)
                # NUM_STEPS: total steps of 'mini_episodes' episodes, 
                # but exclude last steps whose amount is not enough to calculate returns

                # flatten for training
                returns = returns.squeeze(1)  # (NUM_STEPS)
                advantages = advantages.squeeze(1)  # (NUM_STEPS)

                # determine mini_batch_size on the fly
                mini_batch_size = int(len_returns / 20)
            
                self._update_model(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages)

                # delete trained samples
                del l_log_probs[:len_returns]
                del l_actions[:len_returns]
                del l_rewards[:len_returns]
                del l_returns[:len_returns]
                del l_values[:len_returns]
                del l_states[:len_returns]
                del l_masks[:len_returns]

            if episode % 100 == 0 or early_stop:
                self._save_model()
                print(f'Episode: {episode:5} | Reward: {log["episode reward"]:5.1f} | Mean Reward: {mean_train_reward:5.1f}')

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
        if test:
            observation = self._preprocess_state(observation)

        dist, value = self.model(observation)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)

        if test:
            return action.item()
        else:
            return action, log_prob_action, value


    def _preprocess_state(self, state):
        """
        Concatenate two frames along dim=1 as input to the model.
        """
        state = torch.from_numpy(prepro(state)).unsqueeze(0)
        prev_state = self.prev_state if self.prev_state is not None else torch.zeros_like(state)
        x = torch.cat([state, prev_state], dim=1).to(self.device)
        self.prev_state = state
        return x  # (1, (80*80)*2)


    def _save_model(self, save_path='./saved_models/ppo'):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, 'model_ppo.pt'))
        # print(f'Save model to {save_path}')


    def _load_model(self, load_path='./saved_models/ppo'):
        self.model.load_state_dict(torch.load(os.path.join(load_path, 'model_ppo.pt')))
        # print(f'Load model from {load_path}')


    def _compute_returns(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        returns = []
        gae = 0

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])

        return returns


    def _ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantages):
        batch_size = states.shape[0]
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield (states[rand_ids], actions[rand_ids], log_probs[rand_ids],
                   returns[rand_ids], advantages[rand_ids])


    def _update_model(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages,
                      clip_eps=0.2, critic_loss_weight=0.5, entropy_weight=0.01):
        for _ in range(ppo_epochs):
            for state, action, old_log_prob, return_, advantage in self._ppo_iter(
                mini_batch_size, states, actions, log_probs, returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_prob = dist.log_prob(action)

                ratio = (new_log_prob - old_log_prob).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()
                loss = actor_loss + critic_loss_weight * critic_loss - entropy_weight * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()


    def _normalize(self, data):
        return (data - data.mean()) / (data.std() + 1e-8)