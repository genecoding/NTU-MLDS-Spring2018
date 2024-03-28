# Homework 4
* [4-1 Policy Gradient]
* [4-2 Deep Q Learning]

## Note
### 4-1 Policy Gradient
* Implement
  - [x] Policy Gradient
  - [x] PPO (Proximal Policy Optimization), including
    - [x] GAE (Generalized Advantage Estimation)
* Hyperparameters
  * Policy Gradient  
    Set `OUTPUT_DIM` 2 to fasten training.
    ```python
    INPUT_DIM = 80 * 80
    OUTPUT_DIM = 2
    HID_DIM = 256
    learning_rate = 1e-3
    num_episodes = 10000
    ```
  * PPO
    ```python
    INPUT_DIM = (80 * 80) * 2
    OUTPUT_DIM = env.action_space.n (6)
    HID_DIM = 256
    learning_rate = 1e-3
    num_episodes = 10000
    mini_episodes = 10
    mini_steps = 128
    ppo_epochs = 8
    entropy_weight = 0.01
    max_norm = 1.0
    ```
### 4-2 Deep Q Learning
* Implement
  - [x] DQN
  - [x] DDDQN (Dueling Double Deep Q Network), including
    - [x] Double DQN
    - [x] Dueling DQN
* Hyperparameters
  * DQN & DDDQN
    ```python
    INPUT_SHAPE = (4, 84, 84)
    OUTPUT_DIM = env.action_space.n (4)
    HID_DIM = 512
    BUFFER_SIZE = 10000
    learning_rate = 2.5e-4
    num_steps = 5_000_000
    batch_size = 32
    train_freq = 4
    update_freq = 1000
    ```

## Result
* Learing curve
  * Policy Gradient & PPO
    ![re411]
    ![re412]
    ![re413]
  * DQN & DDDQN
    ![re421]
    ![re422]
    ![re423]
* Playing video
  * Policy Gradient
  * PPO
  * DQN
  * DDDQN

## Reference
* http://karpathy.github.io/2016/05/31/rl/
* https://github.com/higgsfield/RL-Adventure
* https://github.com/higgsfield/RL-Adventure-2



[4-1 Policy Gradient]: https://docs.google.com/presentation/d/1bsXDirSx0hS0fJJQU2p1SeTG9ayMN_s_JBP2B8XQoMk
[4-2 Deep Q Learning]: https://docs.google.com/presentation/d/1RlGBmr8WwftbwnnnZm5B4h0emc8v4aGtn-dJomAQJLg
[re411]: result/re411.png
[re412]: result/re412.png
[re413]: result/re413.png
[re421]: result/re421.png
[re422]: result/re422.png
[re423]: result/re423.png
