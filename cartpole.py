import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Create a simple policy network
def create_policy_network(input_dim, output_dim):
    model = tf.keras.Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    return model

# Choose an action according to the policy network output
def choose_action(policy_network, state):
    print(state[0])
    action_probs = policy_network(state[0]).numpy().flatten()
    action = np.random.choice(range(len(action_probs)), p=action_probs)
    return action

# Compute the discounted rewards
def discounted_rewards(rewards, gamma):
    discounted = np.zeros_like(rewards)
    cumulative = 0
    for i in reversed(range(len(rewards))):
        cumulative = cumulative * gamma + rewards[i]
        discounted[i] = cumulative
    return discounted

# Train the policy network using policy gradient
def train_policy_network(policy_network, states, actions, rewards, gamma=0.99):
    discounted_r = discounted_rewards(rewards, gamma)
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)

    with tf.GradientTape() as tape:
        logits = policy_network(states)
        action_masks = tf.one_hot(actions, num_actions)
        log_probs = tf.reduce_sum(action_masks * tf.math.log(logits), axis=1)
        loss = -tf.reduce_sum(log_probs * discounted_r)

    grads = tape.gradient(loss, policy_network.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))

# Main training loop
if __name__ == '__main__':
    env = gym.make('CartPole')
    num_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    policy_network = create_policy_network(state_dim, num_actions)
    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        episode_states, episode_actions, episode_rewards = [], [], []

        while True:
            action = choose_action(policy_network, state)
            next_state, reward, done, _ = env.step(action)

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state

            if done:
                break

        # Train the policy network using the collected data
        states = np.array(episode_states, dtype=np.float32)
        actions = np.array(episode_actions, dtype=np.int32)
        rewards = np.array(episode_rewards, dtype=np.float32)

        train_policy_network(policy_network, states, actions, rewards)

        print(f'Episode {episode + 1}, '
              f'Total reward: {sum(episode_rewards):.2f}, '
              f'Episode length: {len(episode_rewards)}')

