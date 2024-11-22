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
    OUTPUT_DIM = env.action_space.n  # 6
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
    OUTPUT_DIM = env.action_space.n  # 4
    HID_DIM = 512
    BUFFER_SIZE = 10000
    learning_rate = 2.5e-4
    num_steps = 5_000_000
    batch_size = 32
    train_freq = 4
    update_freq = 1000
    ```

## Result
* Learning curves
  * Policy Gradient & PPO  
    (x-axis: episode / y-axis: score)
    ![re411]
    ![re412]
    ![re413]
  * DQN & DDDQN  
    (x-axis: episode / y-axis: clipped score)
    ![re421]
    ![re422]
    ![re423]  
  Surprisingly PPO and DDDQN (especially DDDQN) didn't bring significant improvement in this homework. RL models need to see different observations as many as possible during training to perform well, and there is only one worker/environment in this homework, I guess that's why PPO and DDDQN didn't outperform too much here...

* Playing videos
  | Policy Gradient | PPO       |
  |:---------------:|:---------:|
  |![pong-pg]       |![pong-ppo]|
    
  | DQN           | DDDQN           |
  |:-------------:|:---------------:|
  |![breakout-dqn]|![breakout-dddqn]|
  
## Reference
* http://karpathy.github.io/2016/05/31/rl/
* https://github.com/higgsfield/RL-Adventure
* https://github.com/higgsfield/RL-Adventure-2
* https://huggingface.co/learn/deep-rl-course/unit0/introduction



[4-1 Policy Gradient]: https://docs.google.com/presentation/d/1bsXDirSx0hS0fJJQU2p1SeTG9ayMN_s_JBP2B8XQoMk
[4-2 Deep Q Learning]: https://docs.google.com/presentation/d/1RlGBmr8WwftbwnnnZm5B4h0emc8v4aGtn-dJomAQJLg
[re411]: result/re411.png
[re412]: result/re412.png
[re413]: result/re413.png
[re421]: result/re421.png
[re422]: result/re422.png
[re423]: result/re423.png
[pong-pg]: result/pong-pg-episode-0.gif
[pong-ppo]: result/pong-ppo-episode-0.gif
[breakout-dqn]: result/breakout-dqn-episode-0.gif
[breakout-dddqn]: result/breakout-dddqn-episode-0.gif
