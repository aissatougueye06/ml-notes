# GLOSSAIRE

Les termes rencontrés, définis en trois lignes. Pour répondre vite à « ça veut dire
quoi, déjà ? » — le « comment faire » est dans MEMO.md, le « pourquoi » dans les slides.

## Outillage Python

- **Module** — un fichier `.py`.
- **Paquet** — un dossier de modules avec un `__init__.py`.
- **Linter** (`ruff check`) — cherche des problèmes : imports inutilisés, variables mortes,
conventions. Juge, ne réécrit pas.
- **Formateur** (`ruff format`) — réécrit la mise en forme, ne juge rien. On lance les deux.
- **Mode éditable** (`pip install -e`) — Python pointe vers mes fichiers au lieu d'en copier
une version figée.

## Machine learning

- **Estimateur** — tout objet scikit-learn qui a un `.fit()`. Terme générique :
transformateurs et prédicteurs en sont tous les deux.
- **Transformateur** (*transformer*) — transforme des données en d'autres données.
`fit` / `transform`, jamais `predict`. Ex. : `StandardScaler`, `OneHotEncoder`,
`PolynomialFeatures`, `ColumnTransformer`. À ne pas confondre avec l'architecture
Transformer (attention), sans rapport.
- **Prédicteur** — apprend une relation X → y et produit une prédiction.
`fit` / `predict`, jamais `transform`. Ex. : `LinearRegression`, `LogisticRegression`,
`DummyRegressor`.
- **Pipeline** — objet qui enchaîne préparation et modèle. Un seul `.fit()`, un seul
`.predict()`, un seul fichier sérialisé. Toutes les étapes sauf la dernière sont des
transformateurs ; la dernière est un prédicteur.
- **Sérialiser** — écrire un objet Python dans un fichier ; **désérialiser**, le relire.
`joblib` pour les modèles.
- **Artefact** — fichier produit par le code et régénérable : un modèle `.joblib`. Hors Git.
- **Train/serve skew** — les données ne subissent pas les mêmes transformations à
l'entraînement et en production. Première cause de bug en production.
