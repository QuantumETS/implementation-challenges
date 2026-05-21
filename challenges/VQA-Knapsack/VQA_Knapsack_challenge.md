# Défi - Algorithme variationnel pour résoudre le problème du sac à dos (knapsack problem)

## Objectif

Implémenter un algorithme variationnel quantique capable de résoudre une petite instance du problème du sac à dos, ou *knapsack problem*, avec 8 objets.

Le défi demande de modéliser une fonction de coût à partir d'une instance de knapsack, de construire un circuit quantique paramétré, d'échantillonner des solutions candidates, puis d'utiliser un optimiseur classique pour favoriser les meilleures solutions valides.

L'objectif principal est de produire une solution binaire représentant les objets sélectionnés dans le sac.

---

## Description mathématique du problème

Une instance de knapsack avec 8 objets sera fournie.

Chaque objet `i` possède une valeur $v_i$ et un poids $w_i$.

La capacité maximale du sac est notée $W$.

Une solution est représentée par une chaîne binaire :

$$
\vec{x} = (x_0, x_1, \dots, x_7)
$$

où $x_i=1$ si l'objet `i` est sélectionné, et 0 sinon.

Le but est de maximiser la valeur totale :

$$
V(\vec{x}) = \sum_{i=0}^{7} v_i x_i
$$

sous la contrainte :

$$
\sum_{i=0}^{7} w_i x_i \leq W
$$

Si l'on souhaite transformer le problème en minimisation, on peut inverser le signe de la valeur pour obtenir une fonction de coût à minimiser :

$$
\min_{x\in \{0,1\}^8} C(x) = -\sum_{i=0}^{7} v_i x_i
$$

---

## Instance du problème

L'instance de knapsack à résoudre est la suivante :

```python
values = [8, 5, 6, 9, 4, 7, 3, 10]
weights = [4, 3, 5, 6, 2, 4, 1, 7]
capacity = 12
```

## Astuces pour implémenter la solution

Voici quelques pistes suggérées pour implémenter la solution :

### Modélisation de la fonction de coût

Au problème de **minimisation** de la valeur totale, il faut ajouter une pénalité pour les solutions qui dépassent la capacité du sac. Un terme de pénalité peut être défini comme suit :

$$
P \max\left(0, \sum_{i=0}^{7} w_i x_i - W\right)^2
$$

où `P` est un scalaire positif arbitraire suffisamment grand pour décourager les solutions invalides. Un choix raisonable pour `P` pourrait être la somme totale des valeurs :

$$
P = \sum_{i=0}^{7} v_i
$$

Une fois qu'une fonction de coût incluant la pénalité est définie, elle peut être utilisée pour construire un Hamiltonien quantique représentant le problème de knapsack.

### Construction de l'ansatz

L'Hamiltonien $\hat{H}$ trouvé à partir de la fonction de coût est justement Hamiltonien de coût ou *Cost Hamiltonian*. Pour élaborer un algorithme variationnel, il sera nécessaire de construire un ansatz basé sur cet Hamiltonien, par exemple en utilisant une approche de type QAOA ou un ansatz plus générique.

### Échantillonnage et optimisation

Une fois que l'ansatz est défini, il faut échantillonner des solutions candidates en mesurant le circuit quantique. Les échantillons obtenus peuvent être utilisés pour estimer la valeur de la fonction de coût et guider l'optimisation des paramètres du circuit.

### Comparer avec une solution exacte

Comme l'instance contient seulement 8 objets, il est possible de tester les :

$$
2^8 = 256
$$

solutions possibles par recherche exhaustive.

Cette recherche brute-force permet d'obtenir la meilleure solution valide et de comparer le résultat variationnel avec l'optimum exact.

---

## Résultats suggérés à produire

Les résultats peuvent inclure :

1. une courbe du coût moyen en fonction des itérations
2. une courbe de la meilleure valeur valide trouvée en fonction des itérations
3. un histogramme des bitstrings mesurés à la fin de l'optimisation
4. une comparaison entre la solution variationnelle et la solution brute-force
5. une comparaison entre différentes valeurs du coefficient de pénalité `P`
6. une comparaison entre plusieurs profondeurs d'ansatz.

---

## Critères de réussite

L'objectif minimal est de trouver au moins une solution valide dont la valeur est supérieure à celle d'une sélection aléatoire moyenne.

Un bon résultat devrait retrouver la solution optimale et présenter clairement sa valeur totale, son poids total, et l'écart avec l'optimum exact.

## Ressources supplémentaires

- [Cours sur les algorithmes variationnels](https://quantum.cloud.ibm.com/learning/en/courses/variational-algorithm-design)
- [Modélisation d'un problème combinatoire avec Qiskit](https://quantum.cloud.ibm.com/learning/en/courses/utility-scale-quantum-computing/variational-quantum-algorithms#3-quantum-optimization-with-qiskit-patterns)
- [Exemple d'implémentation QAOA pour une problème de MaxCut](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm)
- [Construction d'ansatz pour QAOA en Qiskit](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.QAOAAnsatz)
