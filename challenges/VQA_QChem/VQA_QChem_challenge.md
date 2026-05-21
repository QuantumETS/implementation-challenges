# Défi - Algorithme variationnel pour approximer l'état fondamental de $OH^-$

## Objectif

Implémenter un algorithme variationnel quantique capable d'approximer l'énergie fondamentale d'un Hamiltonien moléculaire réduit représentant l'ion hydroxyde :

$$
\mathrm{OH^-}
$$

Le défi demande de modéliser une ansatz variationnelle, de modéliser un fonction de coût à partir de l'Hamiltonien fourni, et d'utiliser un optimiseur classique pour trouver les paramètres qui minimisent l'énergie estimée.

---

## Données fournies

L'Hamiltonien du système sera donné sous la forme :

$$
H = \sum_i c_i P_i
$$

où chaque `c_i` est un coefficient réel et chaque `P_i` est une chaîne de Pauli, par exemple :

$$
P_i \in \{I, X, Y, Z\}^{\otimes n}
$$

### Récupérer l'Hamiltonien à partir d'une base de données

Pennylane propose une base de données de molécules avec leurs Hamiltoniens correspondants. Vous pouvez accéder à ces données via l'API de Pennylane, par exemple pour l'ion hydroxyde:

```python
import pennylane as qml
from pennylane.ops.op_math import LinearCombination

hydroxide_dataset = qml.data.load("qchem", molname="OH-", bondlength=0.964, basis="STO-3G", attributes=["hamiltonian", "hf_state", "tapered_hamiltonian"])
```

Il sera ensuite possible d'extraire l'Hamiltonien réduit à partir du dataset pour l'utiliser dans votre algorithme VQE.

```python
hydroxide_hamiltonian = hydroxide_dataset[0].tapered_hamiltonian
```

---

## Résultats suggérés à produire

Les résultats peuvent inclure :

1. une courbe de l'énergie en fonction des itérations
2. une comparaison entre plusieurs optimiseurs
3. une table des meilleurs paramètres trouvés
4. une comparaison entre la meilleure énergie obtenue et une énergie de référence (par exemple, à partir d'une diagonalisation exacte)
5. une analyse qualitative de la convergence.

---

## Critères de réussite

L'objectif minimal est d'obtenir une énergie finale inférieure à l'énergie initiale.

Un bon résultat devrait obtenir une énergie proche de la valeur exacte ou de référence, avec une convergence visible en moins d'une centaine d'itérations.

## Ressources supplémentaires

- [Cours sur les algorithmes variationnels](https://quantum.cloud.ibm.com/learning/en/courses/variational-algorithm-design)
- [Documentation Pennylane sur les Hamiltoniens moléculaires](https://pennylane.ai/datasets/collection/qchem)
- [Exemple d'implémentation VQE pour une molécule simple](https://quantum.cloud.ibm.com/learning/en/modules/computer-science/vqe)
