# Résolution du Problème Frozen Lake avec Q-Learning et SARSA

## Description
Ce projet vise à résoudre le problème de **Frozen Lake**, un environnement classique proposé par [Gymnasium](https://gymnasium.farama.org/environments/toy_text/frozen_lake/), en utilisant les algorithmes d'apprentissage par renforcement **Q-Learning** et **SARSA**.

L'objectif principal est de maximiser les gains dans des scénarios simulant une surface glissante ou non, tout en explorant les différences de performances entre ces deux algorithmes.

---

## Contenu
1. **Introduction :**
   - Utilisation des algorithmes Q-Learning (hors politique) et SARSA (sur politique) pour résoudre Frozen Lake.
   - Tracer et analyser les courbes d'évolution des gains sur plusieurs épisodes.

2. **Environnement Frozen Lake :**
   - **Carte utilisée :** 8x8.
   - **Actions possibles :** Gauche, Droite, Haut, Bas.
   - **Propriétés spécifiques :**
     - `is_slippery=True` : Surface glissante, transitions imprévisibles.
     - `is_slippery=False` : Surface non glissante, transitions déterministes.
   - **Objectif :** Atteindre la case "Goal" sans tomber dans les trous.

3. **Méthodologie :**
   - Évaluation des performances pour deux configurations (`is_slippery=False` et `is_slippery=True`).
   - Variation des hyperparamètres (α, γ, ϵ) pour analyser leur impact.
   - Comparaison des récompenses obtenues par Q-Learning et SARSA.

---

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/AnassamzilUE/frozen-lake-reinforcement-learning.git
   cd frozen-lake-reinforcement-learning
