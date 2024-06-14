# Reinforcement Learning Seminar

This repository hosts code for a seminar on Reinforcement Learning.

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

To configure hyperparameters, please refer to the `./configs/` directory.



