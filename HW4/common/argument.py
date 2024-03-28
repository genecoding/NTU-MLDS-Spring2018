def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    args = parser.parse_args()
    
    if args.train_pg:
        parser.add_argument('--learning_rate', dest='lr', type=float, default=1e-3, help='learning rate for training')
        parser.add_argument('--num_episodes', type=int, default=10_000, help='number of episodes to run the game')
    elif args.train_ppo:
        parser.add_argument('--learning_rate', dest='lr', type=float, default=1e-3, help='learning rate for training')
        parser.add_argument('--num_episodes', type=int, default=10_000, help='number of episodes to run the game')
        parser.add_argument('--mini_episodes', type=int, default=10, help='number of episodes to run for one update')
        parser.add_argument('--mini_steps', type=int, default=128, help='number of steps to run for computing return')
        parser.add_argument('--ppo_epochs', type=int, default=8, help='number of epochs to train the model with sampled data')
        # parser.add_argument('--mini_batch_size', type=int, default=512, help='mini batch size for training')
    elif args.train_dqn:
        parser.add_argument('--learning_rate', dest='lr', type=float, default=2.5e-4, help='learning rate for training')
        parser.add_argument('--num_steps', type=int, default=5_000_000, help='number of steps to run the game')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--train_freq', type=int, default=4, help='frequency to train the model')
        parser.add_argument('--update_freq', type=int, default=1000, help='frequency to synchronize the target model')
    elif args.train_dddqn:
        parser.add_argument('--learning_rate', dest='lr', type=float, default=2.5e-4, help='learning rate for training')
        parser.add_argument('--num_steps', type=int, default=5_000_000, help='number of steps to run the game')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--train_freq', type=int, default=4, help='frequency to train the model')
        parser.add_argument('--update_freq', type=int, default=1000, help='frequency to synchronize the target model')
        
    return parser
