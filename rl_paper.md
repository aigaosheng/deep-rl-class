## Papers on Deep Reinforcement Learning recommmended by HF RL courses

- [Reinforcement Learning: An Introduction, Richard Sutton and Andrew G. Barto Chapter 1, 2 and 3](http://incompleteideas.net/book/RLbook2020.pdf)
- [Foundations of Deep RL Series, L1 MDPs, Exact Solution Methods, Max-ent RL by Pieter Abbeel](https://youtu.be/2GwBez0D20A)
- [Spinning Up RL by OpenAI Part 1: Key concepts of RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

## Gym

- [Getting Started With OpenAI Gym: The Basic Building Blocks](https://blog.paperspace.com/getting-started-with-openai-gym/)
- [Make your own Gym custom environment](https://www.gymlibrary.dev/content/environment_creation/)

## Monte Carlo and TD Learning [[mc-td]]

To dive deeper into Monte Carlo and Temporal Difference Learning:

- <a href="https://stats.stackexchange.com/questions/355820/why-do-temporal-difference-td-methods-have-lower-variance-than-monte-carlo-met">Why do temporal difference (TD) methods have lower variance than Monte Carlo methods?</a>
- <a href="https://stats.stackexchange.com/questions/336974/when-are-monte-carlo-methods-preferred-over-temporal-difference-ones"> When are Monte Carlo methods preferred over temporal difference ones?</a>

## Q-Learning [[q-learning]]

- <a href="http://incompleteideas.net/book/RLbook2020.pdf">Reinforcement Learning: An Introduction, Richard Sutton and Andrew G. Barto Chapter 5, 6 and 7</a>
- <a href="https://youtu.be/Psrhxy88zww">Foundations of Deep RL Series, L2 Deep Q-Learning by Pieter Abbeel</a>

- [Foundations of Deep RL Series, L2 Deep Q-Learning by Pieter Abbeel](https://youtu.be/Psrhxy88zww)
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Double Deep Q-Learning](https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Dueling Deep Q-Learning](https://arxiv.org/abs/1511.06581)

## Introduction to Policy Optimization

- [Part 3: Intro to Policy Optimization - Spinning Up documentation](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)


## Policy Gradient

- [https://johnwlambert.github.io/policy-gradients/](https://johnwlambert.github.io/policy-gradients/)
- [RL - Policy Gradient Explained](https://jonathan-hui.medium.com/rl-policy-gradients-explained-9b13b688b146)
- [Chapter 13, Policy Gradient Methods;  Reinforcement Learning, an introduction by Richard Sutton and Andrew G. Barto](http://incompleteideas.net/book/RLbook2020.pdf)

## Implementation

- [PyTorch Reinforce implementation](https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py)
- [Implementations from DDPG to PPO](https://github.com/MrSyee/pg-is-all-you-need)

# Bonus: Learn to create your own environments with Unity and MLAgents

**You can create your own reinforcement learning environments with Unity and MLAgents**. Using a game engine such as Unity can be intimidating at first, but here are the steps you can take to learn smoothly.

## Step 1: Know how to use Unity

- The best way to learn Unity is to do ["Create with Code" course](https://learn.unity.com/course/create-with-code): it's a series of videos for beginners where **you will create 5 small games with Unity**.

## Step 2: Create the simplest environment with this tutorial

- Then, when you know how to use Unity, you can create your [first basic RL environment using this tutorial](https://github.com/Unity-Technologies/ml-agents/blob/release_20_docs/docs/Learning-Environment-Create-New.md).

## Step 3: Iterate and create nice environments

- Now that you've created your first simple environment you can iterate to more complex ones using the [MLAgents documentation (especially Designing Agents and Agent part)](https://github.com/Unity-Technologies/ml-agents/blob/release_20_docs/docs/)
- In addition, you can take this free course ["Create a hummingbird environment"](https://learn.unity.com/course/ml-agents-hummingbirds) by [Adam Kelly](https://twitter.com/aktwelve)

## Bias-variance tradeoff in Reinforcement Learning

If you want to dive deeper into the question of variance and bias tradeoff in Deep Reinforcement Learning, you can check out these two articles:

- [Making Sense of the Bias / Variance Trade-off in (Deep) Reinforcement Learning](https://blog.mlreview.com/making-sense-of-the-bias-variance-trade-off-in-deep-reinforcement-learning-79cf1e83d565)
- [Bias-variance Tradeoff in Reinforcement Learning](https://www.endtoend.ai/blog/bias-variance-tradeoff-in-reinforcement-learning/)

## Advantage Functions

- [Advantage Functions, SpinningUp RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html?highlight=advantage%20functio#advantage-functions)

## Actor Critic

- [Foundations of Deep RL Series, L3 Policy Gradients and Advantage Estimation by Pieter Abbeel](https://www.youtube.com/watch?v=AKbX1Zvo7r8)
- [A2C Paper: Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783v2)

##  An introduction to multi-agents

- [Multi-agent reinforcement learning: An overview](https://www.dcsc.tudelft.nl/~bdeschutter/pub/rep/10_003.pdf)
- [Multiagent Reinforcement Learning, Marc Lanctot](https://rlss.inria.fr/files/2019/07/RLSS_Multiagent.pdf)
- [Example of a multi-agent environment](https://www.mathworks.com/help/reinforcement-learning/ug/train-3-agents-for-area-coverage.html?s_eid=PSM_15028)
- [A list of different multi-agent environments](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/)
- [Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents](https://bit.ly/3nVK7My)
- [Dealing with Non-Stationarity in Multi-Agent Deep Reinforcement Learning](https://bit.ly/3v7LxaT)

## Self-Play and MA-POCA

- [Self Play Theory and with MLAgents](https://blog.unity.com/technology/training-intelligent-adversaries-using-self-play-with-ml-agents)
- [Training complex behavior with MLAgents](https://blog.unity.com/technology/ml-agents-v20-release-now-supports-training-complex-cooperative-behaviors)
- [MLAgents plays dodgeball](https://blog.unity.com/technology/ml-agents-plays-dodgeball)
- [On the Use and Misuse of Absorbing States in Multi-agent Reinforcement Learning (MA-POCA)](https://arxiv.org/pdf/2111.05992.pdf)

## PPO Explained

- [Towards Delivering a Coherent Self-Contained Explanation of Proximal Policy Optimization by Daniel Bick](https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf)
- [What is the way to understand Proximal Policy Optimization Algorithm in RL?](https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl)
- [Foundations of Deep RL Series, L4 TRPO and PPO by Pieter Abbeel](https://youtu.be/KjWF8VIMGiY)
- [OpenAI PPO Blogpost](https://openai.com/blog/openai-baselines-ppo/)
- [Spinning Up RL PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Paper Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

## PPO Implementation details

- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [Part 1 of 3 — Proximal Policy Optimization Implementation: 11 Core Implementation Details](https://www.youtube.com/watch?v=MEt6rrxH8W4)

## Importance Sampling

- [Importance Sampling Explained](https://youtu.be/C3p2wI4RAi8)

## Toolkit
- [CleanRL is a Deep Reinforcement Learning library that provides high-quality single-file implementation with research-friendly features](https://github.com/vwxyzjn/cleanrl)
- [Stable Baselines3 (SB3) is a set of reliable implementations of reinforcement learning algorithms in PyTorch. ](https://github.com/DLR-RM/stable-baselines3)
- [Gymnasium is an open source Python library for developing and comparing reinforcement learning algorithms](https://github.com/Farama-Foundation/Gymnasium)
- [A library to load and upload Stable-baselines3 models from the Hub with Gymnasium and Gymnasium compatible environment](https://github.com/huggingface/huggingface_sb3)
- [ a free and open-source cross-platform library for the development of multimedia applications like video games using Python](https://github.com/pygame/pygame)
- [Imageio is a mature Python library that makes it easy to read and write image and video data](https://github.com/imageio/imageio)
- [a Python library for video editing: cuts, concatenations, title insertions, video compositing (a.k.a. non-linear editing), video processing, and creation of custom effects.](https://github.com/Zulko/moviepy/)
- [Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. ](https://github.com/optuna/optuna)
- [Set of robotic environments based on PyBullet physics engine and gymnasium.](https://github.com/qgallouedec/panda-gym)
- [DI-engine is a generalized decision intelligence engine for PyTorch and JAX.](https://github.com/opendilab/DI-engine/tree/main)