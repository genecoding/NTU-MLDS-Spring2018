"""

### NOTICE ###
You DO NOT need to upload this file

"""
import argparse
from common.test import test
from common.environment import Environment


def parse():
    parser = argparse.ArgumentParser(description="MLDS 2018 HW4")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
    parser.add_argument('--train_ppo', action='store_true', help='whether train PPO')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--train_dddqn', action='store_true', help='whether train DDDQN')
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_ppo', action='store_true', help='whether test PPO')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--test_dddqn', action='store_true', help='whether test DDDQN')
    try:
        from common.argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name or 'Pong-v0'
        env = Environment(env_name, args)
        from agent_dir.agent_pg import Agent_PG
        agent = Agent_PG(env, args)
        agent.train()
        
    if args.train_ppo:
        env_name = args.env_name or 'Pong-v0'
        env = Environment(env_name, args)
        from agent_dir.agent_ppo import Agent_PPO
        agent = Agent_PPO(env, args)
        agent.train()

    if args.train_dqn:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dir.agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        agent.train()
        
    if args.train_dddqn:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dir.agent_dddqn import Agent_DDDQN
        agent = Agent_DDDQN(env, args)
        agent.train()

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
