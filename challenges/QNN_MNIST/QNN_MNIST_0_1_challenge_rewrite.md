# Défi - Classificateur quantique miniature pour MNIST

## Objectif

Implémenter un réseau de neurones quantique capable de classifier des images MNIST représentant les chiffres `0` et `1`.

Le défi porte sur la construction d'un pipeline complet de classification hybride classique-quantique :

```text
images MNIST -> prétraitement classique -> vecteurs de faible dimension -> encodage quantique -> circuit variationnel -> prédiction -> optimisation
```

La performance finale n'est pas l'objectif principal. L'objectif est plutôt de comprendre comment transformer des images classiques en données compatibles avec un petit circuit quantique, puis d'entraîner ce circuit comme un modèle de classification binaire.

Comme le défi ne suppose pas de connaissances préalables en vision par ordinateur, les étapes de chargement, de filtrage et de réduction des données sont guidées par des extraits de code.

---

## Pourquoi réduire MNIST avant l'entraînement ?

Une image MNIST est une image en niveaux de gris de taille :

$$
28 \times 28 = 784
$$

Chaque image contient donc 784 valeurs de pixels. Un circuit quantique pour l'envergure qui nous intéresse ne peux pas raisonnablement encoder directement 784 caractéristiques, car cela demanderait trop de qubits, trop de portes, ou un encodage trop complexe.

Le but du prétraitement est donc de transformer chaque image en un petit vecteur, par exemple :

$$
x \in \mathbb{R}^4
$$

ou :

$$
x \in \mathbb{R}^8
$$

Chaque composante du vecteur peut ensuite être encodée dans un qubit avec une rotation, par exemple avec un encodage angulaire.

---

## Données fournies

Le jeu de données MNIST peut être chargé avec `torchvision`.

### Charger MNIST

```python
import torch
from torchvision import datasets
from torchvision.transforms import v2 as transforms

# Convertit les images PIL en tenseurs float32 dans l'intervalle [0, 1].
# Pourquoi ? Les pixels MNIST sont initialement des entiers entre 0 et 255.
# Pour l'entraînement, il est plus stable de travailler avec des valeurs normalisées.
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
])

mnist_train = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

mnist_test = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)
```

À ce stade, chaque exemple est une paire :

```text
(image, label)
```

où `image` est un tenseur de forme :

```text
[1, 28, 28]
```

Le `1` représente le canal de couleur. MNIST est en niveaux de gris, donc il n'y a qu'un seul canal.

Pour d'autres exemples sur la manipulation des données, vous pouvez consulter le notebook `notebooks/MNIST-dataset.ipynb` fourni.

---

## Préparer les données pour l'entraînement

### Filtrer les chiffres 0 et 1

Le MNIST original contient les chiffres de `0` à `9`. Pour simplifier le problème, conserver seulement les images dont le label est `0` ou `1`.

```python
def filter_digits(dataset, allowed_digits=(0, 1)):
    images = []
    labels = []

    for image, label in dataset:
        if label in allowed_digits:
            images.append(image)
            labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels

x_train, y_train = filter_digits(mnist_train) # x = images, y = labels
x_test, y_test = filter_digits(mnist_test)


print(f"Filtered training dataset size (0 and 1): {len(x_train)}")
print(f"Filtered test dataset size (0 and 1): {len(x_test)}")
```

La taille du jeu de données filtré devrait être d'environ :

```text
Filtered training dataset size (0 and 1): 12665
Filtered test dataset size (0 and 1): 2115
```

---

### Limiter le nombre d'exemples

Le dataset est encore relativement grand pour un entraînement rapide. Limiter le nombre d'exemples à un petit nombre, par exemple:

```python
n_train = 100
n_test = 40

x_train = x_train[:n_train]
y_train = y_train[:n_train]

x_test = x_test[:n_test]
y_test = y_test[:n_test]
```

---

## Deux chemins possibles pour le prétraitement

Le défi propose deux approches de prétraitement. Les deux ont le même objectif général : réduire une image de 784 pixels vers un petit nombre de caractéristiques compatibles avec un circuit quantique.

### Chemin A - Réduction de dimension par PCA

La PCA transforme chaque image aplatie en un vecteur de faible dimension qui conserve autant que possible les directions de plus grande variance du jeu de données.

Dimension avant PCA :

$$
28 \times 28 = 784
$$

Dimension après PCA, par exemple :

$$
784 \rightarrow 4
$$

ou :

$$
784 \rightarrow 8
$$

Cela signifie que chaque image est résumée par 4 ou 8 pixels synthétiques, qui sont des combinaisons linéaires des pixels originaux.

Pour une implémentation de PCA, vous pouvez utiliser `sklearn.decomposition.PCA` ou `torch.pca_lowrank`, mais l'implémentation avec `sklearn` est plus simple pour ce défi.

#### Préparer les images pour PCA

```python
import numpy as np

# Retire la dimension du canal et aplatit chaque image.
# Forme initiale : [N, 1, 28, 28]
# Forme finale : [N, 784]
x_train_flat = x_train.reshape(len(x_train), -1).numpy()
x_test_flat = x_test.reshape(len(x_test), -1).numpy()
```

PCA attend un tableau de forme [nombre_exemples, nombre_caractéristiques].
Une image 28 x 28 doit donc être transformée en vecteur de longueur 784.

#### Standardiser les pixels

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train_flat)
x_test_scaled = scaler.transform(x_test_flat)
```

La PCA dépend de la variance des caractéristiques.
La standardisation permet d'éviter que certaines caractéristiques dominent seulement à cause de leur échelle numérique.

#### Appliquer PCA

```python
from sklearn.decomposition import PCA
n_features = 4

pca = PCA(n_components=n_features)

x_train_reduced = pca.fit_transform(x_train_scaled)
x_test_reduced = pca.transform(x_test_scaled)
```

La différence entre `fit_transform` et `transform` est que `fit_transform` dans torch est que `fit_transform` applique `fit` avant `transform`, servant à générer des données d'entraînement réduites, tandis que `transform` applique la transformation déjà apprise sur les données de test.

#### Ramener les valeurs dans un intervalle compatible avec des angles

Les rotations quantiques utilisent des angles. Il est donc pratique de transformer les caractéristiques vers un intervalle comme :

$$
[-\pi, \pi]
$$

```python
def scale_to_angle_range(X):
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)

    # Évite une division par zéro si une composante est constante.
    denominator = np.where(X_max - X_min == 0, 1, X_max - X_min)

    X_scaled = 2 * np.pi * (X - X_min) / denominator - np.pi
    return X_scaled

x_train = scale_to_angle_range(x_train_reduced)
x_test = scale_to_angle_range(x_test_reduced)
```

Les valeurs PCA n'ont pas naturellement une échelle d'angle.
Les placer dans [-pi, pi] rend leur utilisation directe dans des portes RY ou RX plus naturelle.

Résumé des transformations du chemin PCA :

```text
Shape of ...
Image MNIST : [1, 28, 28]
Image aplatie : [784]
Vecteur PCA : [4] ou [8]
```

Ainsi, seulement 4 ou 8 qubits sont nécessaire pour encoder les features PCA.

---

### Chemin B - Extraction de caractéristiques par convolution

Une convolution applique de petits filtres sur l'image pour détecter des motifs locaux, par exemple des traits, des transitions ou des courbes. Contrairement à la PCA, qui travaille sur l'image aplatie, une convolution exploite la structure spatiale de l'image.

Dimension initiale :

$$
1 \times 28 \times 28
$$

En appliquant une couche de convolution et une réduction spatiale, on peut extraire des caractéristiques (*features maps*). Par exemple, une convolution avec 4 filtres de taille 3x3 suivie d'un pooling peut produire un vecteur de 4 caractéristiques, ou plus si on utilise plus de filtres.

#### Définir un extracteur convolutionnel simple

```python
import torch.nn as nn

class SimpleConvPreprocessor(nn.Module):
    def __init__(self, n_features=4):
        super().__init__()

        self.features = nn.Sequential(
            # Entrée : [N, 1, 28, 28]
            # Sortie : [N, 4, 26, 26]
            # Pourquoi 26 ? Un noyau 3x3 sans padding réduit 28 à 26.
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3),
            nn.ReLU(),

            # Réduit la résolution spatiale.
            # Sortie : [N, 4, 13, 13]
            nn.MaxPool2d(kernel_size=2),

            # Transforme les cartes de caractéristiques en vecteurs.
            nn.Flatten(),

            # Projette le grand vecteur convolutionnel vers quelques caractéristiques.
            nn.Linear(4 * 13 * 13, n_features),
        )

    def forward(self, x):
        return self.features(x)
```

#### Utiliser l'extracteur convolutionnel

```python
n_features = 4
conv_preprocessor = SimpleConvPreprocessor(n_features=n_features)

with torch.no_grad():
    x_train_features = conv_preprocessor(x_train)
    x_test_features = conv_preprocessor(x_test)

x_train_reduced = x_train_features.numpy()
x_test_reduced = x_test_features.numpy()
```

#### Ramener les caractéristiques dans un intervalle d'angles

```python
x_train = scale_to_angle_range(x_train_reduced)
x_test = scale_to_angle_range(x_test_reduced)
```

Résumé des transformations du chemin convolution :



---

## Comparaison des deux chemins de prétraitement

| Chemin | Transformation principale | Dimension typique | Ce que l'approche accomplit |
|---|---|---:|---|
| PCA | Projection linéaire globale | `784 -> 4` ou `784 -> 8` | Résume les variations principales des images dans un petit vecteur |
| Convolution | Extraction de motifs locaux | `[1, 28, 28] -> [4]` ou `[8]` | Détecte des motifs spatiaux simples avant de produire un vecteur réduit |

Pour ce défi, l'un ou l'autre des chemins est possible.

---

## Sortie du modèle

La prédiction peut être obtenue à partir de l'espérance d'une observable, par exemple :

$$
\langle Z_0 \rangle
$$

La sortie du circuit est alors une valeur réelle entre `-1` et `+1`, compatible avec les labels encodés.

La prédiction finale peut être définie comme :

$$
\hat{y} =
\begin{cases}
1 & \text{si } \langle Z_0 \rangle \geq 0 \\
0 & \text{sinon}
\end{cases}
$$

---

## Fonction de coût

Une fonction de coût simple pour la classification binaire est l'erreur quadratique moyenne :

$$
L(\theta) = \frac{1}{N}\sum_{i=1}^{N} \left(f_\theta(x_i) - y_i\right)^2
$$

où `f_theta(x_i)` est la sortie du circuit pour l'exemple `x_i`, et `y_i` vaut `-1` ou `+1`.

---

## Résultats suggérés à produire

Les résultats peuvent inclure :

1. une courbe de la fonction de coût en fonction des itérations
2. l'accuracy sur l'ensemble d'entraînement
3. l'accuracy sur l'ensemble de test
4. une comparaison entre plusieurs profondeurs d'ansatz
5. une comparaison entre le chemin PCA et le chemin convolution

---

## Critères de réussite

L'objectif minimal est d'obtenir une accuracy supérieure au hasard, c'est-à-dire supérieure à 50% sur l'ensemble de test.

Une solution complète devrait :

1. charger MNIST
2. filtrer les chiffres `0` et `1`
3. limiter le nombre d'exemples
4. transformer les labels vers `-1` et `+1`
5. réduire les images vers 4 ou 8 caractéristiques avec PCA ou convolution
6. encoder ces caractéristiques dans un circuit quantique
7. ajouter un ansatz variationnel
8. entraîner les paramètres avec un optimiseur classique
9. évaluer l'accuracy finale sur un ensemble de test.

Un bon résultat devrait obtenir une accuracy nettement supérieure au hasard sur l'ensemble de test, supérieur à 90%.

La solution devrait aussi expliquer les choix principaux du modèle : méthode de réduction de dimension, nombre de qubits, encodage des données, ansatz utilisée et optimiseur choisi.
