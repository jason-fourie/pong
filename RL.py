import numpy as np
from numpy.random import choice
import tensorflow as tf
import random
import cv2 #Helps read in pixel data
import pong # Pong class creates a game of pong
from collections import deque #Use queue data structure.
import os

#Hyper parameters
TRAIN = True
SEED = 1984

ACTIONS = 3 # move up, move down, or do nothing

GAMMA = 0.99 # Learning rate

# Values for epsilon-greedy method
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
# Number of steps over which to anneal epsilon from initial to final
EXPLORE = 500000 # Train until halted
OBSERVE = 50000

# How many entries to store in the queue
REPLAY_MEMORY = 20000 # Don't overload memory
# Batch size for training
BATCH = 100

WEIGHTS_DIRECTORY = 'checkpoints'

# Create tensorflow graph
def createGraph():

    # Convolutional layer and bias vector

    W_conv1 = tf.Variable(tf.truncated_normal([6, 6, 4, 32], stddev=0.02), dtype='float32')
    b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))

    W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.02), dtype='float32')
    b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))

    W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.02), dtype='float32')
    b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]))

    W_fc4 = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.02), dtype='float32')
    b_fc4 = tf.Variable(tf.constant(0.01, shape=[512]))

    W_fc5 = tf.Variable(tf.truncated_normal([512, ACTIONS], stddev=0.02), dtype='float32')
    b_fc5 = tf.Variable(tf.constant(0.01, shape=[ACTIONS]))

    # Input for pixel data
    pixel_data = tf.placeholder("float", [None, 60, 60, 4])

    # Computes rectified linear unit activation fucntion on  a 2-D convolution given 4-D input and filter tensors. and
    conv1 = tf.nn.relu(tf.nn.conv2d(pixel_data, W_conv1, strides=[1, 4, 4, 1], padding="SAME") + b_conv1)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, strides=[1, 2, 2, 1], padding="SAME") + b_conv2)

    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding="SAME") + b_conv3)

    conv3_flat = tf.reshape(conv3, [-1, 1024])

    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)

    # Perform dropout when training
    if TRAIN:
        fc4 = tf.nn.dropout(fc4, 0.5, seed=SEED)

    fc5 = tf.matmul(fc4, W_fc5) + b_fc5

    return pixel_data, fc5


# Train deep Q network
def trainGraph(inp, out, sess):

    # To calculate the argmax, multiply the predicted output with aone-hot
    argmax = tf.placeholder("float", [None, ACTIONS])
    gt = tf.placeholder("float", [None]) #ground truth

    action = tf.reduce_sum(tf.matmul(out, tf.transpose(argmax)))
    # Define cost function as squares error of difference between prediction and gt
    cost = tf.reduce_mean(tf.square(action - gt))
    # Define optimization function
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # Initialize the game
    game = pong.PongGame()

    # Create a queue data structure to store history
    D = deque()

    # Obtain current frame
    frame = game.getPresentFrame()
    # Convert RGB colour values to gray-scale
    frame = cv2.cvtColor(cv2.resize(frame, (60, 60)), cv2.COLOR_BGR2GRAY)
    # Push colour values to either white or black
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    # Stack frames to be used as input tensor
    inp_t = np.stack((frame, frame, frame, frame), axis = 2)

    # Create saver directory
    if not os.path.exists(WEIGHTS_DIRECTORY): os.makedirs(WEIGHTS_DIRECTORY)
    save_path = WEIGHTS_DIRECTORY + '/Weights.ckpt'

    # Define tensorflow saver
    saver = tf.train.Saver(tf.global_variables())

    init = tf.global_variables_initializer()
    sess.run(init)

    '''checkpoint = tf.train.latest_checkpoint('./checkpoints')
    
    if checkpoint != None:
        print('Restore Checkpoint %s' % (checkpoint))
        saver.restore(sess, checkpoint)
        print("Model restored.")
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("Initialized new Graph")'''

    t = 0
    epsilon = INITIAL_EPSILON

    # Train model
    while(True):
        # Evaluate output tensor
        out_t = out.eval(feed_dict = {inp : [inp_t]})[0]
        # Initialize argmax function
        argmax_t = np.zeros([ACTIONS])

        # Use epsilon to determine whether to choose actions according to listed probability, or from choice from NN
        if(random.random() <= epsilon):
            maxIndex = choice((0, 1, 2), 1, p=(0.90, 0.05, 0.05))
        else:
            maxIndex = np.argmax(out_t)
        argmax_t[maxIndex] = 1

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # Allocate reward for action
        reward_t, frame = game.getNextFrame(argmax_t)

        # Get pixel data
        frame = cv2.cvtColor(cv2.resize(frame, (60, 60)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (60, 60, 1))
        # Calculate new input tensor
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis = 2)

        # Append input tensor, argmax tensor, reward and updated input tensor to queue of past experiences
        D.append((inp_t, argmax_t, reward_t, inp_t1))

        # Do not more given number of memories
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # Training iteration
        if t > OBSERVE:

            # Select minibatch from history
            minibatch = random.sample(D, BATCH)

            inp_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            inp_t1_batch = [d[3] for d in minibatch]

            gt_batch = []
            out_batch = out.eval(feed_dict = {inp : inp_t1_batch})

            # Add values to batch
            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))

            # Train optimizer
            train_step.run(feed_dict = {
                           gt : gt_batch,
                           argmax : argmax_batch,
                           inp : inp_batch
                           })

        # Update input tensor for use in the next time-step
        inp_t = inp_t1
        t = t+1

        # Save and print current status
        if t % 100000 == 0:
            saver.save(sess, save_path=save_path, global_step = t)

        print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", maxIndex, "/ REWARD", reward_t, "/ Q_MAX %e" % np.max(out_t))


def main():
    # Create session
    sess = tf.InteractiveSession()
    # Get input and output layer by creating the graph
    inp, out = createGraph()
    # Train the graph on input and output in the session
    trainGraph(inp, out, sess)

if __name__ == "__main__":
    main()
