# Challenge d'implémentation

Ce répertoire contient des énoncés de défis d'implémentation pour tester vos compétences en développement d'algorithmes quantiques.

## Structure du projet

Les énoncés de défis sont organisés dans `/challenges`, dans un format Markdown. Chaque énoncé contient une description du défi, des instructions pour l'implémentation et des suggestions de résultats à produire.

## Première configuration

> [!IMPORTANT]
> La configuration actuelle est prévue pour `uv`. Utilisez `uv sync` pour installer les dépendances et `uv run` pour exécuter les commandes du projet.

### Configuration avec [`uv`](https://docs.astral.sh/uv/)

<details>
<summary>Instructions pour uv</summary>

#### 1) Installer les dépendances

```bash
uv sync --group dev
```

#### 2) Lancer les hooks qualité

```bash
uv tool install pre-commit
uvx pre-commit install
uvx pre-commit run --all-files
```

#### 3) Exécuter les tests

```bash
uv run pytest tests/
```

#### 4) Lancer le script d'exemple

```bash
uv run python src/main.py
```

> [!TIP]
> Si vous préférez ne pas installer `pre-commit` globalement, vous pouvez remplacer `uvx pre-commit ...` par une commande équivalente dans un environnement où `pre-commit` est déjà installé.

</details>

## Ressources supplémentaires

### Outils de développement

- [Documentation pytest](https://docs.pytest.org/en/stable/)
- [Documentation pre-commit](https://pre-commit.com/)
- [Documentation Ruff](https://docs.astral.sh/ruff/)
- [Documentation pre-commit Ruff hooks](https://github.com/astral-sh/ruff-pre-commit)
- [PEP 8 - Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

### Gestion d'environnement

- [uv - Documentation](https://docs.astral.sh/uv/)
