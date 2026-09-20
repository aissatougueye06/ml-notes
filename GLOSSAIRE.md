# GLOSSAIRE

Les termes rencontrés, definis en trois lignes. Pour repondre vite a « ca veut dire
quoi, deja ? » — le « comment faire » est dans MEMO.md, le « pourquoi » dans les slides.

## Outillage Python

**Module** — un fichier `.py`.
**Paquet** — un dossier de modules avec un `__init__.py`.
**Linter** (`ruff check`) — cherche des problemes : imports inutilises, variables mortes,
conventions. Juge, ne reecrit pas.
**Formateur** (`ruff format`) — reecrit la mise en forme, ne juge rien. On lance les deux.
**Mode editable** (`pip install -e`) — Python pointe vers mes fichiers au lieu d'en copier
une version figee.

## Machine learning

**Pipeline** — objet qui enchaine preparation et modele. Un seul `.fit()`, un seul
`.predict()`, un seul fichier serialise.
**Serialiser** — ecrire un objet Python dans un fichier ; **deserialiser**, le relire.
`joblib` pour les modeles.
**Artefact** — fichier produit par le code et regenerable : un modele `.joblib`. Hors Git.
**Train/serve skew** — les donnees ne subissent pas les memes transformations a
l'entrainement et en production. Premiere cause de bug en production.