# Reinforcement Learning Seminar

This repository hosts code for a seminar on Reinforcement Learning. It includes simple implementations of Q-Table, Deep Q-Network, Double Deep Q-Network, and REINFORCE on Gym environments.
To configure hyperparameters, please refer to the `./configs/` directory.



To train and evaluate the Q-Table, run:

```bash
python3 run_q.py --env <environment>
```

To train and evaluate the DQN and DDQN, run:

```bash
python3 run_dq.py --env <environment> --agent <dqn or ddqn>
```

To train the policy gradient REINFORCE, run:

```bash
python3 run_pg.py --env <environment>
```

To access real-time plots:

```bash
tensorboard --logdir ./logs/
```



