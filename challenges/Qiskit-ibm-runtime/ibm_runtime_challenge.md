# Défi - Exécuter un programme sur IBM Quantum Platform

## Objectif

L'objectif de ce défi est d'exécuter un programme quantique sur la plateforme IBM Quantum en utilisant les librairies `qiskit` et `qiskit-ibm-runtime`. Il sera demandé d'exécuter un circuit quantique préconçu sur un simulateur bruité configuré à partir d'un modèle de bruit, puis de le faire tourner sur une véritable machine.

Vous devrez comparer les résultats obtenus sur le simulateur et la machine réelle, et expliquer les différences observées.

Une étape bonus consistera à appliquer des techniques de mitigation du bruit pour améliorer les résultats obtenus sur la machine réelle, et à comparer les résultats avant et après mitigation.

## Données fournies

Un circuit quantique préconçu se trouve dans le dossier `notebooks`, accompagné d'exemples de code qui devront être complétés pour exécuter le circuit sur les différentes plateformes.

## Résultats à produire

1. (Optionel) L'histogramme des résultats obtenus à partir du simulateur idéal (sans bruit).
2. L'histogramme des résultats obtenus à partir du simulateur bruité.
3. L'histogramme des résultats obtenus à partir de la machine réelle.
4. (Optionel) Une analyse textuelle comparative des résultats obtenus avec les différentes plateformes, en mettant en évidence les différences et en formulant une hypothèse sur les causes de ces différences (par exemple, les erreurs de mesure, les erreurs de porte, etc.).
5. (Optionel) L'application de techniques de mitigation du bruit pour améliorer les résultats obtenus sur la machine réelle, et une comparaison des résultats avant et après mitigation à l'aide d'histogrammes.
