# Pacman Imitation Learning

Projet d'apprentissage supervisé pour entraîner un agent Pacman à imiter un expert en utilisant un réseau de neurones MLP.

## =Ë Vue d'ensemble

Ce projet implémente un système d'**imitation learning** où un réseau de neurones apprend à jouer à Pacman en imitant les actions d'un expert. Le réseau prend en entrée l'état du jeu (23 features) et prédit l'action à prendre (NORTH, SOUTH, EAST, WEST, STOP).

### Résultats

- **Accuracy sur le test set** : ~87-88%
- **Architecture** : MLP (23  256  128  64  5)
- **Dataset** : 15,018 exemples d'états-actions de l'expert
- **Temps d'entraînement** : ~150 epochs

## <× Architecture du réseau

```
Input Layer:     23 features
                  
Hidden Layer 1:  256 neurones (Linear  BatchNorm  ReLU  Dropout 0.3)
                  
Hidden Layer 2:  128 neurones (Linear  BatchNorm  ReLU  Dropout 0.3)
                  
Hidden Layer 3:  64 neurones  (Linear  BatchNorm  ReLU  Dropout 0.3)
                  
Output Layer:    5 neurones   (Linear, pas d'activation)
                  
                [NORTH, SOUTH, EAST, WEST, STOP]
```

**Total de paramètres** : 47,168 poids

## =Ê Features (23 au total)

### Position de Pacman (2)
- Position X normalisée
- Position Y normalisée

### Information sur le fantôme (4)
- Direction X vers le fantôme
- Direction Y vers le fantôme
- Distance Manhattan au fantôme
- Fantôme adjacent (1 si distance d 1)

### Information sur la nourriture (4)
- Nombre de pastilles restantes
- Direction X vers la nourriture la plus proche
- Direction Y vers la nourriture la plus proche
- Distance à la nourriture la plus proche

### Géométrie du labyrinthe (5)
- Distance au mur nord
- Distance au mur sud
- Distance au mur est
- Distance au mur ouest
- Est un coin (1 si d2 directions légales)

### Niveau de danger (3)
- Niveau de danger (1 / distance_fantôme)
- Fantôme bloque la nourriture
- Options d'échappement (nombre de directions / 4)

### Actions légales (5)
- NORTH légal
- SOUTH légal
- EAST légal
- WEST légal
- STOP légal

## = Installation

### Prérequis

```bash
Python 3.8+
PyTorch 1.10+
```

### Installation des dépendances

```bash
pip install -r requirements.txt
```

Contenu de `requirements.txt`:
```
torch>=1.10.0
pandas>=1.3.0
numpy>=1.21.0
```

## =Á Structure du projet

```
pacman_imitation_learning/
 datasets/
    pacman_dataset.pkl      # Dataset d'entraînement (15,018 exemples)
    pacman_test.pkl         # Dataset de test pour Gradescope

 pacman_module/              # Moteur de jeu (NE PAS MODIFIER)
    game.py
    pacman.py
    layout.py
    ...

 data.py                     # Feature engineering + Dataset
 architecture.py             # Réseau de neurones MLP
 train.py                    # Script d'entraînement
 pacmanagent.py              # Agent qui joue avec le modèle
 run.py                      # Visualisation du jeu
 write_submission.py         # Génération du CSV pour Gradescope

 pacman_model.pth            # Modèle entraîné (sauvegardé après training)
 submission.csv              # Prédictions pour Gradescope

 requirements.txt            # Dépendances Python
 README.md                   # Ce fichier
 explication.txt             # Documentation détaillée (français)
```

## <® Utilisation

### 1. Entraîner le modèle

```bash
python train.py
```

Cela va :
- Charger le dataset (15,018 exemples)
- Split train/test 80/20
- Entraîner pendant 150 epochs
- Sauvegarder le meilleur modèle dans `pacman_model.pth`

**Hyperparamètres** :
- `batch_size = 256`
- `learning_rate = 8e-4`
- `epochs = 150`
- `dropout = 0.3`
- `optimizer = Adam`

### 2. Visualiser l'agent jouer

```bash
python run.py
```

Cela lance le jeu avec interface graphique et l'agent utilise le modèle entraîné.

### 3. Générer le fichier de soumission

```bash
python write_submission.py
```

Cela génère `submission.csv` avec les prédictions pour chaque état du test set.

## >à Détails techniques

### Feature Engineering

Toutes les features sont **normalisées** entre 0 et 1 pour une meilleure convergence :
- Positions : divisées par les dimensions du labyrinthe
- Distances : divisées par la distance Manhattan maximale
- Flags binaires : 0 ou 1

### Loss Function

**CrossEntropyLoss** : loss standard pour la classification multi-classe

```python
Loss = -log(P(action_correcte))
```

Combine automatiquement :
- Softmax (conversion en probabilités)
- Log
- Negative Log Likelihood
- Moyenne sur le batch

### Optimizer

**Adam** (Adaptive Moment Estimation) :
- Adapte le learning rate pour chaque poids individuellement
- Utilise le momentum pour une convergence plus rapide
- Plus stable que SGD classique

### Régularisation

- **BatchNorm** : normalise les activations entre couches
- **Dropout (p=0.3)** : désactive 30% des neurones aléatoirement pendant l'entraînement
- **Early stopping** : sauvegarde le meilleur modèle basé sur test accuracy

## =È Résultats d'entraînement typiques

```
Epoch   1:  Accuracy H 60%   (démarrage aléatoire)
Epoch  50:  Accuracy H 82%
Epoch 100:  Accuracy H 86%
Epoch 120:  Accuracy H 87.5%  Meilleur modèle
Epoch 150:  Accuracy H 87.2% (légère baisse = overfitting)
```

**Note** : L'accuracy plafonne à ~87-88% car dans certaines situations, plusieurs actions sont également valides. Le réseau peut choisir une action différente de l'expert mais tout aussi bonne.

## =' Personnalisation

### Modifier l'architecture

Dans `architecture.py`, vous pouvez modifier :

```python
PacmanNetwork(
    input_features=23,
    num_actions=5,
    hidden_dims=[256, 128, 64],  #  Modifier les dimensions des couches
    activation=nn.ReLU(),         #  Changer l'activation (GELU, LeakyReLU, etc.)
    dropout=0.3                   #  Ajuster le dropout
)
```

### Modifier les hyperparamètres

Dans `train.py`, vous pouvez modifier :

```python
batch_size = 256        # Taille du batch
epochs = 150            # Nombre d'epochs
learning_rate = 8e-4    # Learning rate
test_ratio = 0.2        # Ratio train/test
```

### Ajouter de nouvelles features

Dans `data.py`, fonction `state_to_tensor()` :

```python
def state_to_tensor(state):
    # Ajouter vos nouvelles features ici
    my_feature = calculate_my_feature(state)

    features = [
        # ... features existantes ...
        my_feature,  #  Nouvelle feature
    ]

    return torch.tensor(features, dtype=torch.float32)
```

  **Attention** : Si vous ajoutez des features, modifiez aussi `input_features` dans `architecture.py` !

## = Debugging

### Le modèle ne converge pas

- **Vérifier la normalisation** : toutes les features doivent être ~[0, 1]
- **Réduire le learning rate** : essayer `lr = 5e-4` ou `lr = 3e-4`
- **Augmenter le nombre d'epochs** : essayer 200-250 epochs

### Accuracy très basse (~50-60%)

- **Vérifier le dataset** : le fichier `pacman_dataset.pkl` est-il correct ?
- **Vérifier les features** : sont-elles bien calculées ?
- **Vérifier le mapping actions** : `ACTION_TO_INDEX` est-il correct ?

### Overfitting (train accuracy >> test accuracy)

- **Augmenter le dropout** : essayer `dropout = 0.4` ou `dropout = 0.5`
- **Réduire la taille du réseau** : essayer `hidden_dims=[128, 64, 32]`
- **Ajouter plus de données** : augmenter le dataset si possible

### Le jeu ne s'affiche pas

- **Vérifier tkinter** : `sudo apt-get install python3-tk` (Linux)
- **Utiliser run.py** : `python run.py` pour lancer avec interface graphique

## =Ú Documentation complète

Pour une explication détaillée de chaque étape, consulter **[explication.txt](explication.txt)** :

- PARTIE 0 : Comment fonctionne un réseau de neurones (12 étapes détaillées)
- PARTIE 1 : Vue d'ensemble du projet
- PARTIE 2 : Structure des fichiers
- PARTIE 3 : Feature engineering (data.py)
- PARTIE 4 : Architecture du réseau (architecture.py)
- PARTIE 5 : Entraînement (train.py)
- PARTIE 6 : Génération du CSV (write_submission.py)
- PARTIE 7 : FAQ (Questions fréquentes)

## S FAQ

### Pourquoi un MLP et pas un CNN ?

Les CNN sont faits pour les images 2D où les pixels voisins sont liés. Notre input est un vecteur 1D de 23 features sans relation spatiale. Un MLP est le bon choix.

### Pourquoi normaliser les features ?

Les réseaux de neurones convergent mieux quand toutes les features sont dans la même échelle (~[0, 1]). Sans normalisation, les grandes valeurs domineraient les petites.

### Pourquoi ReLU au lieu de GELU ?

ReLU est plus simple et rapide : `f(x) = max(0, x)`. GELU est plus smooth mais ReLU fonctionne très bien pour ce problème.

### Pourquoi 150 epochs ?

Optimisé empiriquement. Le meilleur modèle est généralement vers epoch 120-130. Plus d'epochs (200+) mène à l'overfitting.

### Est-ce que le modèle fonctionne sur différents labyrinthes ?

Oui ! Les features sont adaptatives (normalisées par la taille du labyrinthe). Le modèle a appris des patterns généraux (fuir les fantômes, aller vers la nourriture) pas juste à mémoriser un labyrinthe.

## = Ressources

- **PyTorch Documentation** : https://pytorch.org/docs/
- **MLP Tutorial** : https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
- **Cross Entropy Loss** : https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- **Adam Optimizer** : https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

## =Ý License

Ce projet est fourni à des fins éducatives.

## =e Auteurs

Projet réalisé dans le cadre du cours d'Intelligence Artificielle.

---

**Bon courage pour le projet ! <®>**
