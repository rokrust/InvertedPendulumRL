import tensorflow as tf
import numpy as np
import random
import gym
env = gym.make('CartPole-v0')

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 500
display_step = 1

# Network Parameters
n_hidden_1 = 120
n_hidden_2 = 120
n_input = 4
n_classes = 2

# tf Graph input
x = tf.placeholder("float32", [None, n_input], name='InputState')
y_0 = tf.placeholder("float32", [None, 1], name='Labels_0')


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    with tf.name_scope('FirstLayer'):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # layer_1 = tf.nn.dropout(layer_1, keep_prob=0.8)

    with tf.name_scope('SecondLayer'):
        # Hidden layer with RELU activation
        layer_2 = tf.matmul(layer_1, weights['h2']) + biases['b2']
        layer_2 = tf.nn.relu(layer_2)
        # layer_2 = tf.nn.dropout(layer_2, keep_prob=0.8)

    with tf.name_scope('OutputLayer'):
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.05), name='WeightsLayer1'),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.05), name='WeightsLayer2'),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=0.05), name='WeightsOutput')
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='BiasLayer1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='BiasLayer2'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='BiasOutput')
}

# Construct model
network = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost_0 = tf.reduce_mean(tf.squared_difference(network[0,1], [y_0], name='SquaredDifference'), name='CostFunction')
optimizer_0 = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95, name='Optimizer').minimize(cost_0)

cost_1 = tf.reduce_mean(tf.squared_difference(network[0,0] , [y_0], name='SquaredDifference'), name='CostFunction')
optimizer_1 = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95, name='Optimizer').minimize(cost_1)

current_state = tf.Variable(tf.zeros([1, 4]), dtype=tf.float32, name='CurrentState')
next_state = tf.Variable(tf.zeros([1, 4]), dtype=tf.float32, name='NextState')  # init model

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("output", sess.graph)

    n_runs = 11000
    n_batch = 1
    # hyper parameters
    epsilon = 1
    gamma = 0.5

    # state variables

    for epoch in range(n_runs):
        if epoch > 3000:
            PRINT_FLAG = True;
        else:
            PRINT_FLAG = False

        D = []
        for batch in range(n_batch):
            if epoch%100 == 0:
                print("Epoch ", epoch)
            next_state = np.reshape(env.reset(), [-1, 4])
            reward = 0  # Final reward
            done = 0  # Termination condition

            while not done:

                current_state = next_state
                [Q] = sess.run(network, feed_dict={x: current_state})

                #Take a random action with probability epsilon
                if random.random() < epsilon:
                    action = np.random.randint(2)

                #Otherwise, choose action with largest q-value
                elif Q[0] > Q[1]:
                    action = 0

                else:
                    action = 1

                # Step through model and update state
                next_state, current_reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [-1, 4])

                reward += current_reward

                if done:
                    D.append((current_state, reward, action, []))
                    if PRINT_FLAG:
                        print("Reward: ", reward)
                    
                else:
                    D.append((current_state, 0.0, action, next_state))

                if PRINT_FLAG:
                    env.render()
                    print("Q: ", Q)
                    print("Action: ", action)
        if epsilon > 0.1:
            epsilon *= 0.99

        random.shuffle(D)  # Pick a random transition from D
        for transitions in D:
            s, r, a, s_n = transitions
            Q_target = [0]
            # Find best Q-value of the next state
            if len(s_n) == 0:   # If terminal state

                #[Q_target] = sess.run(network, feed_dict={x: s})
                Q_target[0] = r
                #Q_target = np.reshape(Q_target, [-1, 2])
                Q_target = np.reshape(Q_target, [-1, 1])
            else:
                [Q_n] = sess.run(network, feed_dict={x: s_n})

                # Get new reward in the updated network
                #[Q_target] = sess.run(network, feed_dict={x: s})

                # Find Q_target values.
                # The non-optimal action is set to have an error of zero
                Q_target[0] = r + gamma * max(Q_n)
                #Q_target = np.reshape(Q_target, [-1, 2])
                Q_target = np.reshape(Q_target, [-1, 1])

            # Train the network
            if a == 0:
                sess.run(optimizer_0, feed_dict={x: s, y_0: Q_target})
            else:
                sess.run(optimizer_1, feed_dict={x: s, y_0: Q_target})


        del D[:]
    writer.close()

# .............................Not used...................................#
"""
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.	
        total_batch = int(mnist.train.num_examples/batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                  "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
"""
# .........................................................................#
