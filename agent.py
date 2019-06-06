
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from collections import deque
import random

class DQN_model:

    def __init__(self, network_name, n_inputs, n_outputs, n_hidden_units):

        """
            Arguments
                - network_name
                - n_inputs : the number of input dimensions
                - n_outputs : the number of input dimensions
                - n_hidden_units : num of hidden units per layer (should be passed as list)
        """

        self.network_name = network_name
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_units = n_hidden_units

    def network(self, trainable = True):

        """
            Organize Q-value function approximation block
        """

        network_name = self.network_name
        n_inputs = self.n_inputs
        n_outputs = self.n_outputs
        n_hidden_units = self.n_hidden_units

        initializer = tf.contrib.layers.variance_scaling_initializer()
        with tf.variable_scope(network_name):

            model = Sequential(name=network_name)
            model.add(Dense(units=n_hidden_units[0],
                            kernel_initializer=initializer, input_dim=n_inputs))
            model.add(Activation('relu'))
            for n_units in n_hidden_units[1:]:
                model.add(Dense(units=n_units, kernel_initializer=initializer))
                model.add(Activation('relu'))
            model.add(Dense(units=n_outputs, kernel_initializer=initializer))
            model.trainable = trainable

            return model

class Agent:

    def __init__(self, sess, n_inputs, n_outputs, n_hidden_units, discount_rate, learning_rate):

        """
            Arguments
                - sess : concurrent tensorflow session
                - n_inputs : the number of input dimensions
                - n_outputs : the number of input dimensions
                - n_hidden_units : num of hidden units per layer (should be passed as list)
                - discount rate
                - learning rate
        """

        self.sess = sess
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_units = n_hidden_units

        # online dqn과 target dqn 생성 (target dqn은 freeze)
        self.online_dqn = DQN_model('online_dqn', n_inputs, n_outputs,
                                                          n_hidden_units).network(trainable=True)
        self.target_dqn = DQN_model('target_dqn', n_inputs, n_outputs,
                                                          n_hidden_units).network(trainable=False)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.memory = deque(maxlen=5000)
        self._compute_loss(discount_rate, learning_rate)
        self._create_summaries()

    def copy_online_to_target(self):

        """
            Copy online-dqn network weights to target-dqn weights
        """

        self.target_dqn.set_weights(self.online_dqn.get_weights())

    def _create_summaries(self):

        """
            Collection of summary operations
        """

        with tf.name_scope('summaries'):
            tf.summary.scalar('loss',self.loss)
            tf.summary.histogram('histogram_loss',self.loss)
            for i in range(self.n_outputs):
                tf.summary.scalar('action {}'.format(i),self.online_dqn_output[0][i])
            summary_op = tf.summary.merge_all()

            self.summary_op = summary_op

    def _compute_loss(self, discount_rate, learning_rate):

        """
            Organize the tensorflow graph given online-dqn q-values and target-dqn q-values
            to compute loss
        """

        X_current_state = tf.placeholder(dtype=tf.float32, shape=(None,self.n_inputs), name='X_current_state')
        X_action = tf.placeholder(dtype=tf.int32, shape=(None,), name='X_action')
        X_reward = tf.placeholder(dtype=tf.float32, shape=(None,), name='X_reward')
        X_next_state = tf.placeholder(dtype=tf.float32, shape=(None,self.n_inputs), name='X_next_state')
        X_continues = tf.placeholder(dtype=tf.float32, shape=(None,), name='X_continues')

        target_dqn_output = self.target_dqn(X_next_state)
        max_target_q_value = tf.reduce_max(target_dqn_output, axis=1, keepdims=True, name='max_target_q_value')
        y_val = tf.add(X_reward, tf.multiply(discount_rate, X_continues * max_target_q_value),name='y_val')

        online_dqn_output = self.online_dqn(X_current_state)
        online_q_value = tf.reduce_sum(online_dqn_output * tf.one_hot(X_action, self.n_outputs),
                                       axis=1, keepdims=True, name='online_q_value')

        error = tf.abs(y_val - online_q_value, name='error')
        loss = tf.reduce_mean(tf.square(error), name='loss')
        feed_list = [X_current_state, X_action, X_reward, X_next_state, X_continues]

        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss,global_step=self.global_step)

        self.feed_list = feed_list
        self.online_dqn_output = online_dqn_output
        self.training_op = training_op
        self.loss = loss

    def train(self, batch):

        """
             Run the session with training operation to update the network
        """

        sess = self.sess
        loss = self.loss
        feed_list = self.feed_list
        training_op = self.training_op
        summary_op = self.summary_op

        current_state = np.array([tup[0] for tup in batch])
        action = np.array([tup[1] for tup in batch])
        reward = np.array([tup[2] for tup in batch])
        next_state = np.array([tup[3] for tup in batch])
        continues = np.array([tup[4] for tup in batch])

        feed_dict = {feed: input for feed, input in
                     zip(feed_list, [current_state,action,reward,next_state,continues])}

        loss_val, _, summary = sess.run([loss,training_op,summary_op],feed_dict=feed_dict)

        return loss_val, summary


    def act(self, eps, output, random_seed=0):

        """
            Choose action based on calculated q-values with eps (epsilon)
        """

        if np.random.random_sample() < eps:
            return np.random.randint(self.n_outputs)
        else:
            return np.argmax(output)

    def sample_from_the_memory(self, batch_size):

        """
            Random sampling of size n (batch size) from experience stored memory
        """

        batch = random.sample(self.memory, batch_size)
        return batch

    def save_in_the_memory(self,state,action,reward,next_state,done):

        """
            Save experience to the memory
        """

        self.memory.append((state,action,reward,next_state,done))