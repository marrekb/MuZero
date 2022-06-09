# MuZero

Hey, this repository is oriented to my implementation of the MuZero agent (MZ) and Monte Carlo tree search (MCTS). During my studies, I tried to find the fast available implementation of MZ. However, there were a few slow versions on Github. It is understandable because as you probably know, building a decision tree by the MCTS approach in each state is computationally intensive. Of course, the agent needs to collect as much data as possible, therefore he plays multi games simultaneously. For each running game is necessary to create new MCTS (in each step). 

However, I got the **idea to implement MCTS by tensors**. This implementation is able to **process a batch of observations on GPU**. It is faster than building MCTS via CPU and processes each observation separately. Now, we have finished the paper about the implementation of the proposed MCTS, so you have to wait :). In folder "proposed_mcts" are my implementation (GPU.py) and another popular implementations. You can check my tuned version of my MCTS in 'core/mcts.py'.

## Experiments in Pong environment

I tested MZ with my version of MCTS in the Atari domain - environment Pong. MZ played 20 games simultaneously in the cycle without multiprocessing (because the replay buffer took a lot of memory and my RAM was almost completely full ~ 32GB). For training, I used one NVIDIA GeForce RTX 2080 Ti. One training lasts about 1 day (I know, too long, but it is MuZero). 
There are two training with a different models in the plot. I tried to use as much deeper a model as possible. Maybe you would be able to train Pong with a smaller model. Just try it, good luck! :)

![alt text](https://github.com/marrekb/MuZero/blob/main/plot_pong.png?raw=true)

Next goal is to train MZ in game Breakout. 
