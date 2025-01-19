import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run_sarsa(episodes, is_training=False, render=False):
    # Changement pour la carte 8x8
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None,max_episode_steps=200)
    
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open('frozen_lake8x8_sarsa.pkl', 'rb') as f:
            q = pickle.load(f)

    # Hyperparamètres ajustés pour 8x8
    learning_rate_a = 0.3  # Réduit car plus d'états
    discount_factor_g = 0.9  # Augmenté car chemin plus long
    epsilon = 1
    min_epsilon = 0.01
    # Ajusté pour une décroissance plus lente car environnement plus complexe
    epsilon_decay_rate = 0.00005 
    
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)
    
    # Ajout du suivi des échecs et succès
    success_rate = []
    window_size = 1000  # Fenêtre pour calculer le taux de succès
    
    for i in range(episodes):
        state = env.reset()[0]
        terminated = truncated = False
        
        # Sélection de l'action initiale avec une exploration guidée
        if rng.random() < epsilon:
            if np.max(q[state]) > 0:  # Si on a déjà appris quelque chose
                # 50% chance de choisir parmi les meilleures actions
                best_actions = np.where(q[state] == np.max(q[state]))[0]
                if rng.random() < 0.5:
                    action = rng.choice(best_actions)
                else:
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()
        else:
            action = np.argmax(q[state, :])
            
        steps = 0
        #max_steps = 200  # Limite de pas pour éviter les épisodes trop longs
            
        while not (terminated or truncated) :#and steps < max_steps:
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            # Même stratégie pour la prochaine action
            if rng.random() < epsilon:
                if np.max(q[new_state]) > 0:
                    best_actions = np.where(q[new_state] == np.max(q[new_state]))[0]
                    if rng.random() < 0.5:
                        next_action = rng.choice(best_actions)
                    else:
                        next_action = env.action_space.sample()
                else:
                    next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q[new_state, :])
            
            # Mise à jour SARSA avec bonus pour les états peu visités
            q[state, action] += learning_rate_a * (
                reward + discount_factor_g * q[new_state, next_action] - q[state, action]
            )
            
            state = new_state
            action = next_action
            rewards_per_episode[i] += reward
            steps += 1
        
        # Décroissance adaptative d'epsilon
        if i < episodes * 0.5:  # Exploration plus longue (70% des épisodes)
            epsilon = max(min_epsilon, 1.0 - i * epsilon_decay_rate)
        else:
            epsilon = max(min_epsilon, epsilon * 0.9999)
            
        # Calcul du taux de succès sur la fenêtre
        if i % window_size == 0 and i > 0:
            success_rate.append(np.mean(rewards_per_episode[i-window_size:i]))
            
        if i % 100 == 0:
            print(f"Episode {i}, Total Reward: {np.sum(rewards_per_episode[:i+1])}")
            print(f"epsilon: {epsilon:.3f}")

    # Visualisation des récompenses
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    
    plt.figure(figsize=(12, 6))
    plt.plot(sum_rewards)
    plt.title('SARSA - FrozenLake 8x8')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards (last 100 avg)')
    plt.savefig('frozen_lake8x8_sarsa.png')
    
    if is_training:
        with open("frozen_lake8x8_sarsa.pkl", "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    # Augmentation du nombre d'épisodes pour l'environnement plus complexe
    run_sarsa(50000, is_training=True)
    #run_sarsa(20,is_training=False,render=True)
