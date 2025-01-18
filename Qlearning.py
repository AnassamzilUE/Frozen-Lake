import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

# Fonction principale pour exécuter l'entraînement ou le test avec Q-learning
def run(episodes, is_training=False, render=False):
    # Création de l'environnement FrozenLake 8x8 avec ou sans surface glissante
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None)

    # Initialisation ou chargement de la Q-table
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))  # Initialisation d'une Q-table de taille 64x4
    else:
        with open('frozen_lake8x8.pkl', 'rb') as f:
            q = pickle.load(f)

    # Hyperparamètres du Q-learning
    learning_rate_a = 0.5  # Taux d'apprentissage (α)
    discount_factor_g = 0.95  # Facteur de réduction (γ), favorise les récompenses futures
    epsilon = 1  # Exploration initiale à 100%
    epsilon_decay_rate = 0.0001  # Taux de décroissance de l'exploration
    rng = np.random.default_rng()  # Générateur de nombres aléatoires

    rewards_per_episode = np.zeros(episodes)  # Stocke les récompenses cumulées pour chaque épisode

    for i in range(episodes):
        state = env.reset()[0]  # Réinitialisation de l'environnement, état initial
        terminated, truncated = False, False

        while not terminated and not truncated:
            # Politique epsilon-greedy pour choisir une action
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # Exploration
            else:
                action = np.argmax(q[state, :])  # Exploitation

            # Transition vers un nouvel état
            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                # Mise à jour de la Q-table selon l'équation Q-learning
                q[state, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state  # Mise à jour de l'état courant

        # Réduction progressive d'epsilon pour diminuer l'exploration
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Ajustement du taux d'apprentissage lorsque epsilon atteint 0
        if epsilon == 0:
            learning_rate_a = 0.0001

        # Enregistrement des récompenses
        if reward == 1:
            rewards_per_episode[i] = 1
        print(f"Episode {i}, Total Reward: {np.sum(rewards_per_episode[:i+1])}")
        print(f"epsilon: {epsilon}")

    env.close()

    # Calcul des récompenses moyennes sur 100 épisodes pour visualisation
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.title('Qlearning - FrozenLake')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards (last 100 avg)')
    plt.savefig('frozen_lake8x8.png')

    # Sauvegarde de la Q-table après entraînement
    if is_training:
        with open("frozen_lake8x8.pkl", "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    run(20000, is_training=True)  # Entraînement avec 20,000 épisodes
