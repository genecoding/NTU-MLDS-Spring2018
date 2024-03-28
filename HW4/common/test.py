"""

### NOTICE ###
You DO NOT need to upload this file

"""

import argparse
import numpy as np
from common.environment import Environment

seed = 11037

def parse():
    parser = argparse.ArgumentParser(description="MLDS 2018 HW4")
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_ppo', action='store_true', help='whether test PPO')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--test_dddqn', action='store_true', help='whether test DDDQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(seed)
    for i in range(total_episodes):
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        #playing one game
        while(not done):
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))


def run(args):
    if args.test_pg:
        env = Environment('Pong-v0', args, test=True)
        from agent_dir.agent_pg import Agent_PG
        agent = Agent_PG(env, args)
        test(agent, env)
        
    if args.test_ppo:
        env = Environment('Pong-v0', args, test=True)
        from agent_dir.agent_ppo import Agent_PPO
        agent = Agent_PPO(env, args)
        test(agent, env)

    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dir.agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)

    if args.test_dddqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dir.agent_dddqn import Agent_DDDQN
        agent = Agent_DDDQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)
