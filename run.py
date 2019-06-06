
# python 3

from env import Market_environmet
from agent import Agent

import numpy as np
import pandas as pd
import tensorflow as tf

if __name__ == '__main__':

    # Basic parameter setting
    cash_in_hand = 1000
    done = False
    mode = 'test'

    # Architecture setting
    n_inputs = 4
    n_outputs = 3
    n_hidden_units = [32,32]

    # log path setting
    checkpoint_path = './data/dqn_model'
    graph_data_path = './data'
    summary_path = './logs'

    # Hyperparameter setting
    n_steps = 200000
    training_start_steps = 2000
    copy_steps = 5
    show_steps = 1000
    save_steps = 5000
    discount_rate = 0.99
    learning_rate = 5e-5
    batch_size = 128

    eps_max = 1
    eps_min = 0.1
    eps_decay = 1 / n_steps
    eps = eps_max
    eps_test = 0

    data = pd.read_csv('data.csv',index_col=False).values
    split_index = int(len(data) * 0.8)
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    train_mean = np.mean(train_data, axis=0, keepdims=False)
    train_std = np.std(train_data, axis=0, keepdims=False)
    norm_train_data = (train_data - train_mean) / train_std  
    norm_test_data = (test_data - train_mean) / train_std

    stock_price_mean = train_mean[1]
    stock_price_std = train_std[1]

    tf.set_random_seed(seed=0)

    if mode == 'train':
        with tf.Session() as sess:
            env = Market_environmet(norm_train_data, n_outputs, cash_in_hand, stock_price_mean, stock_price_std)
            agent = Agent(sess, n_inputs, n_outputs,
                          n_hidden_units=n_hidden_units,
                          discount_rate=discount_rate,
                          learning_rate=learning_rate)

            saver = tf.train.Saver()
            writer = tf.summary.FileWriter(summary_path,sess.graph)

            if tf.train.latest_checkpoint(graph_data_path):
                saver.restore(sess, tf.train.latest_checkpoint(graph_data_path))
                print('training model successfully loaded')
                current_step = 0
                training_step = agent.global_step.eval()
                print('loaded from global step {}'.format(training_step))
            else:
                current_step = 0
                training_step = 0
                tf.global_variables_initializer().run()

            current_state = env._reset()
            loss_list = []
            while True:
                if current_step > training_start_steps and current_step % show_steps == 0:
                    print('current step : {}'.format(current_step))

                # If current step exceeds over n_step, then terminate
                if training_step > n_steps:
                    break
                else:
                    current_step += 1

                # If current_state == done, then env reset
                if done:
                    current_state = env._reset()

                # Derive the action
                output = agent.online_dqn.predict(np.reshape(current_state,newshape=(1,-1)))
                action = agent.act(eps, output, random_seed=current_step)

                # Execute the derived action in env
                next_state, reward, done, info = env._step(action)

                # Save env information to replay memory
                agent.memory.append((current_state, action, reward, next_state, 1 - done))

                # current state update
                current_state = next_state

                # For experience exploring purpose
                if current_step < training_start_steps:
                    continue

                # Train
                batch = agent.sample_from_the_memory(batch_size)
                loss, summary = agent.train(batch)
                training_step = agent.global_step.eval()
                writer.add_summary(summary, global_step=training_step)

                # average loss check
                if training_step % show_steps == 0:
                    average_loss_val = np.mean(np.array(loss_list), keepdims=False)
                    print('training step : {} / average_loss : {}'.
                          format(training_step, average_loss_val))
                    loss_list.clear()
                else:
                    loss_list.append(loss)

                # copy
                if current_step % copy_steps == 0:
                    agent.copy_online_to_target()

                # eps update
                eps = np.max([eps_min,eps - eps_decay])

                # save the model
                if current_step % save_steps == 0:
                    saver.save(sess, checkpoint_path, global_step=agent.global_step)
    else:
        with tf.Session() as sess:
            env = Market_environmet(norm_test_data, n_outputs, cash_in_hand, stock_price_mean, stock_price_std)
            agent = Agent(sess,
                          n_inputs=n_inputs,
                          n_outputs=n_outputs,
                          n_hidden_units=n_hidden_units,
                          discount_rate=discount_rate,
                          learning_rate=learning_rate)

            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(graph_data_path))

            current_state = env._reset()
            value_portfolio = []
            q_values_portfolio = []
            action_list = []

            while True:
                # If current_state == done then break, else append to portfolio
                if done:
                    break
                else:
                    value_portfolio.append([current_state[0],current_state[1],current_state[3]])

                # Derive action
                output = agent.online_dqn.predict(np.reshape(current_state, newshape=(1,-1)))
                action = agent.act(eps=eps_test, output=output)
                action_list.append(action)
                q_values_portfolio.append(output[0])

                # Execute the derived action in env
                next_state, reward, done, info = env._step(action)

                # current state update
                current_state = next_state