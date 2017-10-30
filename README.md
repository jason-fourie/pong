# pong
Implementation of Q-learning algorithm to teach agent to play pong

## Training Script: RL.py
The training script contains two methods and a collection of hyper parameters. The first method is createGraph(), which is used to initialise the requisite computational graph of a convolutional neural network. The second method is trainGraph(), which takes the preprocessed frame from a gaming window and its image under the CNN as input, and updates the weights of the model accordingly. 

## Game Script: pong.py
The game script creates a virtual gaming window and creates the objects necessary to play pong. Methods for updating their positions and allocating scores to the agent are included.
