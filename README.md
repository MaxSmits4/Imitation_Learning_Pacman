# Pacman Imitation Learning

Projet d'apprentissage supervisé pour entraîner un agent Pacman à imiter un expert via un réseau de neurones MLP.

## Vue d'ensemble

Ce projet implémente un système d'imitation learning où un réseau de neurones apprend à jouer à Pacman en imitant les actions d'un expert. Le réseau prend en entrée l'état du jeu (32 features) et prédit l'action à effectuer parmi 5 possibles : NORTH, SOUTH, EAST, WEST, STOP.

**Résultats obtenus :**
- Accuracy sur le test set : ~87-88%
- Architecture : MLP (32 → 128 → 64 → 32 → 5)
- Dataset : 15 018 exemples d'états-actions
- Entraînement : ~150 epochs

## Installation

**Prérequis :** Python 3.8+, PyTorch 2.0+

```bash
pip install -r requirements.txt
```

## Utilisation

**Entraîner le modèle :**
```bash
python train.py
```

**Visualiser l'agent jouer :**
```bash
python run.py
```

**Générer le fichier de soumission :**
```bash
python write_submission.py
```

## Structure du projet

```
pacman_imitation_learning/
├── datasets/
│   ├── pacman_dataset.pkl      # Dataset d'entraînement
│   └── pacman_test.pkl         # Dataset de test
├── pacman_module/              # Moteur de jeu (ne pas modifier)
├── data.py                     # Feature engineering + Dataset
├── architecture.py             # Réseau de neurones MLP
├── train.py                    # Script d'entraînement
├── pacmanagent.py              # Agent utilisant le modèle
├── run.py                      # Visualisation du jeu
├── write_submission.py         # Génération du CSV
├── pacman_model.pth            # Modèle entraîné
└── submission.csv              # Prédictions pour Gradescope
```

## Architecture du réseau

```
Input (32)  →  Linear+BN+GELU+Dropout (128)
            →  Linear+BN+GELU+Dropout (64)
            →  Linear+BN+GELU+Dropout (32)
            →  Linear (5)
```

Chaque couche cachée : Linear → BatchNorm → GELU → Dropout (0.3)

## Features (32)

| Catégorie | Nb | Description |
|-----------|:--:|-------------|
| Position | 2 | X et Y normalisées de Pacman |
| Ghost | 7 | Direction X/Y vers fantôme, distance Manhattan, flags directionnels (N/S/E/W) |
| Food | 9 | Nb pastilles, distance moyenne, direction et distance vers la plus proche, flags directionnels |
| Maze | 9 | Distance aux 4 murs, distance au centre X/Y, flags coin/couloir/espace ouvert |
| Legal | 5 | Flags pour chaque action légale |

Toutes les features sont normalisées (positions par dimensions du labyrinthe, distances par distance Manhattan maximale).

## Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| Batch size | 128 |
| Learning rate | 1e-3 |
| Epochs | 150 |
| Dropout | 0.3 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Split train/test | 80/20 |

## Personnalisation

**Modifier l'architecture** (dans `architecture.py`) :
```python
PacmanNetwork(
    input_features=32,
    num_actions=5,
    hidden_dims=[128, 64, 32],
    activation=nn.GELU(),
    dropout=0.3
)
```

**Ajouter des features** (dans `data.py`) : modifier `state_to_tensor()` et mettre à jour `input_features` dans `architecture.py`.

## Troubleshooting

| Problème | Solution |
|----------|----------|
| Pas de convergence | Vérifier la normalisation, réduire le learning rate |
| Accuracy ~50-60% | Vérifier le dataset et le mapping des actions |
| Overfitting | Augmenter le dropout, réduire la taille du réseau |
| Pas d'affichage | Installer tkinter (`sudo apt-get install python3-tk`) |

## FAQ

**Pourquoi un MLP et pas un CNN ?**
L'input est un vecteur 1D de features extraites manuellement, sans structure spatiale 2D. Un MLP est adapté pour ce type de données tabulaires.

**Pourquoi GELU ?**
GELU est une activation plus smooth que ReLU, standard dans les architectures modernes (BERT, GPT). Elle pondère les inputs par leur valeur plutôt que de les filtrer par leur signe.

**Pourquoi normaliser ?**
Les réseaux convergent mieux quand toutes les features sont dans la même échelle. Sans normalisation, les grandes valeurs dominent le gradient.

**Pourquoi l'accuracy plafonne à ~88% ?**
Dans certaines situations, plusieurs actions sont également valides. Le réseau peut prédire différemment de l'expert tout en étant correct.

**Le modèle généralise-t-il ?**
Oui. Les features sont normalisées par la taille du labyrinthe, donc le modèle apprend des patterns généraux (fuir les fantômes, aller vers la nourriture).

## Documentation

Pour une explication détaillée, voir `explication.txt`.

## Références

- [1] Ioffe & Szegedy (2015) - Batch Normalization
- [2] Hendrycks & Gimpel (2016) - GELU
- [3] Srivastava et al. (2014) - Dropout
- [4] Kingma & Ba (2015) - Adam Optimizer
- [5] LeCun et al. (1998) - MNIST (principes de classification)

Voir `Bibliography.txt` pour les citations complètes.