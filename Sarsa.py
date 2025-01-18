import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Fonction principale pour exécuter SARSA
def run_sarsa(episodes, is_training=False, render=False):
    # Création de l'environnement FrozenLake 4x4 avec surface glissante
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human' if render else None)

    # Initialisation ou chargement de la Q-table
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))  # Q-table de taille 16x4
    else:
        with open('frozen_lake8x8_sarsa.pkl', 'rb') as f:
            q = pickle.load(f)

    # Hyperparamètres de SARSA
    learning_rate_a = 0.9  # Taux d'apprentissage (α)
    discount_factor_g = 0.9  # Facteur de réduction (γ)
    epsilon = 1  # Exploration initiale à 100%
    epsilon_decay_rate = 0.00001  # Taux de décroissance de l'exploration
    rng = np.random.default_rng()  # Générateur de nombres aléatoires
    rewards_per_episode = np.zeros(episodes)  # Suivi des récompenses par épisode

    for i in range(episodes):
        state = env.reset()[0]  # Réinitialisation de l'état initial
        terminated, truncated = False, False

        # Choix de la première action selon epsilon-greedy
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[state, :])

        while not terminated and not truncated:
            # Transition vers le nouvel état
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Choix de la prochaine action avec epsilon-greedy
            if rng.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q[new_state, :])

            # Mise à jour de la Q-table selon l'équation SARSA
            q[state, action] += learning_rate_a * (
                reward + discount_factor_g * q[new_state, next_action] - q[state, action]
            )

            state = new_state  # Mise à jour de l'état courant
            action = next_action  # Mise à jour de l'action courante

            rewards_per_episode[i] += reward  # Accumulation des récompenses

        # Réduction progressive d'epsilon
        epsilon = max(epsilon - epsilon_decay_rate, 0.1)  # Limite minimale d'epsilon

        print(f"Episode {i}, Total Reward: {np.sum(rewards_per_episode[:i+1])}")
        print(f"epsilon: {epsilon}")

    # Calcul des récompenses moyennes sur 100 épisodes
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.title('SARSA - FrozenLake')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards (last 100 avg)')
    plt.savefig('frozen_lake8x8_sarsa.png')

    # Sauvegarde de la Q-table après entraînement
    if is_training:
        with open("frozen_lake8x8_sarsa.pkl", "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    run_sarsa(50000, is_training=True)  # Entraînement avec 50,000 épisodes
