from __future__ import print_function

import tensorflow as tf
import random
import gym


# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 5
n_hidden_2 = 3
n_input = 4
n_classes = 2

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.matmul(layer_1, weights['h2']) + biases['b2']
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
network = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.squared_difference(labels=y))#tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

	n_runs = 1000
	
	#hyper parameters
	epsilon = 1
	learning_rate = 0.001
	gamma = 0.01

	#state variables	
	current_state = None
	next_state = env.reset() #init model
	reward = 0 #Final reward
	done = 0  #Termination condition

	for range(n_runs):
		D = []
		
		while not done:
			current_state = next_state
			Q_l, Q_r = sess.run(network, feed_dict={x : current_state})

			if random.random() < epsilon:
				action = randint(0, 2)
				epsilon *= 0.7

			elif Q_l > Q_r:
				Q = Q_l
				action = 0

			else:
				Q = Q_r
				action = 1
			
			#Step through model
			next_state, current_reward, done, info = env.step(action)
			reward += current_reward
			
			#Save transitions
			D.append((current_state, Q, action, next_state))
	
		shuffle(D) #Pick a random transition from D
		for transitions in D
			s, r, a, s_n = transitions

			#Find best Q-value of the next state
			Q_n = sess.run(network, feed_dict={x : s_n})
			Q_opt = r + gamma*max(Q_n)
			
			#Find Q_target values. 
			#The non-optimal action is set to have an error of zero
			Q_target = sess.run(network, feed_dict={x : s}) #Get new reward in the updated network
			Q_target[a] = Q_opt
			
			#Train the network
			sess.run(optimizer, feed_dict={x : s, y : Q_target}) #Hopefully this is correct

		del D[:]
	

#.............................Not used...................................#
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
#.........................................................................#
