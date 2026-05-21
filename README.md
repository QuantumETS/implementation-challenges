# Challenge d'implémentation

Ce répertoire contient des énoncés de défis d'implémentation pour tester vos compétences en développement d'algorithmes quantiques.

## Structure du projet

Les énoncés de défis sont organisés dans `/challenges`, dans un format Markdown. Chaque énoncé contient une description du défi, des instructions pour l'implémentation et des suggestions de résultats à produire.

## Première configuration

> [!IMPORTANT]
> La configuration actuelle est prévue pour `uv`. Utilisez `uv sync` pour installer les dépendances et `uv run` pour exécuter les commandes du projet.

### Installation avec [`uv`](https://docs.astral.sh/uv/)

Pour installer les dépendances de tous les défis, utilisez la commande suivante à la racine du projet :

```bash
uv sync --all-packages
```

Pour installer les dépendances d'un défi spécifique, utilisez la commande suivante en remplaçant le nom du défi par celui que vous souhaitez :

```bash
uv sync --package <nom_du_défi>
```

## Ressources supplémentaires

### Outils de développement

- [Documentation pytest](https://docs.pytest.org/en/stable/)
- [Documentation pre-commit](https://pre-commit.com/)
- [Documentation Ruff](https://docs.astral.sh/ruff/)
- [Documentation pre-commit Ruff hooks](https://github.com/astral-sh/ruff-pre-commit)
- [PEP 8 - Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

### Gestion d'environnement

- [uv - Documentation](https://docs.astral.sh/uv/)
