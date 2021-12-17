# exploration-enhanced-reinforcement-learning

## DQN (baseline)
It is the deep-Q-learning algorithm introduced in section Baseline Algorithms in the report.


## Self Curiocity DQN
In this problem, the mouse will receive a reward 1 if he reaches the target otherwise he will receive no reward 0. The reward is really sparse and he will receive 0 reward in almost all his trails which makes DQN unsuitable for this environment.
To solve this problem, we give a curiosity reward to the mouse in order to encourage him go to different state. The curiosity reward is defined as follows:

 1. If he reaches the state he has already arrive, give a -0.4 reward.
 2. If he try to hit the wall of the maze, give a -0.7 reward.
 3. If he reaches the state he has never been to, give a -0.1 reward.

We can create a 7*7 matrix to record whether the mouse has gone to the state during this episode till now.


## Policy Curiocity
In some environment, the state space may be very large and it is unwise to record all the states he has gone to . In this case, we use policy curiosity methods. We give a bonus to the policy which is quite different from others.
The key of this method is how to measure the different between the policy and what the bonus should be. We design two mechanism for DQN baseline and ES baseline and test them on maze environment.

1. Div_DQN:
  (1)	Difference measure between two policies
  (2)	Bonus add to policy

2. Div_ES:
  (1) Difference measure between two policies
  (2) Bonus add to policy

