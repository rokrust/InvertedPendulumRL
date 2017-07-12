import tensorflow as tf
import numpy as np
import random
import gym
env = gym.make('CartPole-v0')

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 10
n_hidden_2 = 10
n_input = 4
n_classes = 2

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
pred = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.matmul(layer_1, weights['h2']) + biases['b2']
    layer_2 = tf.nn.relu(layer_2)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
network = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    n_runs = 1000
    n_batch = 10
    # hyper parameters
    epsilon = 1
    learning_rate = 0.001
    gamma = 0.01

    # state variables
    current_state = None

    for epoch in range(n_runs):
        for batch in range(n_batch):
            print("Epoch ", epoch)
            D = []
            next_state = env.reset()  # init model
            next_state = np.reshape(next_state, [-1, 4])
            reward = 0  # Final reward
            done = 0  # Termination condition
            while not done:
                env.render()
                current_state = next_state
                [Q] = sess.run(network, feed_dict={x: current_state})
                print("\tOUTPUT: ", Q)
                if random.random() < epsilon:
                    action = np.random.randint(2)

                elif Q[0] > Q[1]:
                    action = 0

                else:
                    action = 1

                # Step through model
                print("Action: ", action)
                next_state, current_reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [-1, 4])

                reward += current_reward

                if done:
                    D.append((current_state, reward*100, action, next_state))
                    print("\t\tREWARD: ", reward)
                # Save transitions
                else:
                    D.append((current_state, Q[action], action, next_state))

        if epsilon > 0.1:
            epsilon *= 0.95

        random.shuffle(D)  # Pick a random transition from D
        for transitions in D:
            s, r, a, s_n = transitions

            # Find best Q-value of the next state
            [Q_n] = sess.run(network, feed_dict={x: s_n})
            Q_opt = r + gamma * max(Q_n)

            # Find Q_target values.
            # The non-optimal action is set to have an error of zero
            [Q_target] = sess.run(network, feed_dict={x: s})  # Get new reward in the updated network
            Q_target[a] = Q_opt
            Q_target = np.reshape(Q_target, [-1, 2])
            # Train the network
            sess.run(optimizer, feed_dict={x: s, y: Q_target})  # Hopefully this is correct

        del D[:]
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
