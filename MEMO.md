# MEMO — carnet technique

Carnet personnel, alimenté au fil des projets et du cours
[Machine Learning Specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/).

**Ce qui va ici** : les lignes que je retape sans arrêt, et les messages d'erreur avec
leur cause. **Ce qui ne va pas ici** : les principes et le « pourquoi » — ils sont dans
`memo-machine-learning.pptx`, à relire d'un bloc.

**Règle d'alimentation** : dès qu'un problème m'a coûté plus de dix minutes, il gagne
trois lignes ici, au moment où il arrive. Même chose pour une **décision** tranchée
dans un projet (une version, une convention) : elle est reportée ici, sinon le carnet
diverge du code.

**Sommaire** — [1. Messages d'erreur](#1-messages-derreur--cause--correctif) ·
[2. Mise en place d'un projet](#2-mise-en-place-dun-projet) ·
[3. Pandas](#3-pandas--gestes-courants) · [4. NumPy](#4-numpy--gestes-courants) ·
[5. Machine learning](#5-machine-learning--le-socle) ·
[6. Hygiène de notebook](#6-hygiène-de-notebook) ·
[7. Algorithmique](#7-algorithmique--repères) ·
[8. Méthode d'analyse](#8-méthode-danalyse) ·
[9. Du notebook au module](#9-du-notebook-au-module)

---

## 1. Messages d'erreur → cause → correctif

Indexés par le **texte exact** du message : c'est ce que j'aurai sous les yeux, pas la cause.

Cette section s'enrichit de moins en moins vite : plus le travail avance, moins les
erreurs plantent. Les problèmes de modélisation ne lèvent aucune exception — ils
donnent un chiffre faux. Ils sont donc en §3, §4 (pièges) et §8 (méthode).

### `zsh: command not found: python`
Normal sur macOS : seul `python3` est reconnu hors environnement virtuel.
→ Utiliser `python3` et `pip3`. Dans un `.venv` activé, `python` et `pip` suffisent.

### `ModuleNotFoundError: No module named 'X'` alors que X est installé
Le mauvais environnement est actif. `(.venv)` s'affiche pareil pour tous les projets.
→ `which python` pour voir quel Python répond, `pip list` pour voir ce qu'il contient.
→ Réactiver le bon : `source .venv/bin/activate` depuis le dossier du projet.

### `requires the ipykernel package` (VS Code, notebook)
Le pont entre le `.venv` et l'interface notebook manque.
→ `pip install ipykernel` dans l'environnement activé.

### Le `.venv` n'apparaît pas dans « Select a Python Environment »
VS Code explore le dossier ouvert. Si le dossier parent est ouvert, il ne voit pas le bon.
→ Ouvrir le dossier **du projet** : `cd <projet> && code .`
→ Sinon, rafraîchir la liste (icône ↻) ou `Cmd+Shift+P` → *Python: Select Interpreter*.

### `UnicodeDecodeError` ou accents en charabia (`Ã©`) à la lecture d'un CSV
Encodage non-UTF-8, fréquent sur les exports administratifs français.
→ `pd.read_csv(..., encoding="latin-1")`

### Le CSV se charge en **une seule colonne**
Séparateur point-virgule (convention française, la virgule servant de décimale).
→ `pd.read_csv(..., sep=";")`

### `KeyError` sur une valeur pourtant visible dans `value_counts()`
Espaces de fin invisibles à l'affichage (export à format fixe). `"GO "` ≠ `"GO"`.
→ `df["col"] = df["col"].str.strip()` dès le chargement.
→ Diagnostic : afficher `resultat.index` — les espaces s'y voient entre quotes.

### `Expected 2D array, got 1D array instead` (scikit-learn)
`X` doit être de forme `(n_échantillons, n_variables)`.
→ `x.reshape(-1, 1)` pour une seule variable.

### `The truth value of an array with more than one element is ambiguous`
`and` / `or` / `not` utilisés sur un masque NumPy ou Pandas.
→ `&`, `|`, `~` — **avec parenthèses** autour de chaque condition.

### `operands could not be broadcast together with shapes (3,4) (3,)`
Alignement à droite : 4 vs 3, incompatible.
→ `(3,)` s'applique aux **colonnes**, `(3,1)` aux **lignes**.
→ `.reshape(3, 1)`, ou `agg(..., keepdims=True)` si ça vient d'une agrégation.

### `Cannot perform reduction 'mean' with string dtype`
`df.mean()` sur un DataFrame contenant des colonnes texte.
→ `df.mean(numeric_only=True)`, ou sélectionner les colonnes numériques.

### `RuntimeWarning: overflow encountered` puis `nan` partout (descente de gradient)
Taux d'apprentissage trop grand : le coût augmente au lieu de diminuer.
→ Diviser α, ou normaliser `x` (ce qui élargit beaucoup la plage d'α utilisable).
→ Un `nan` contamine toutes les itérations suivantes : inutile de laisser tourner.
→ Si la divergence est **attendue** (démonstration d'un α trop grand), la déclarer
plutôt que de la subir : `with np.errstate(over="ignore", invalid="ignore"):`. La trace
numpy fait apparaître un chemin local et un PID d'ipykernel qui change à chaque
redémarrage — le notebook ressort modifié dans `git status` sans raison.

### `nan` inattendu dans un résultat, sans erreur
Souvent une variable réutilisée d'une cellule précédente (état fantôme de notebook).
→ *Kernel → Restart and Run All*. Vérifier que les numéros `In[ ]` sont dans l'ordre.

### `cannot reshape array of size 12 into shape (5,3)`
`reshape` ne crée ni ne détruit de données : lignes × colonnes doit égaler le total.

### `ModuleNotFoundError: No module named '<mon_paquet>'` en lançant un script
`pytest` trouve le paquet (il ajoute la racine du projet au chemin de recherche),
`python scripts/x.py` non.
→ `pip install -e .` — c'est exactement ce qu'il répare.

### `UnboundLocalError: cannot access local variable 'X'`
Une variable locale porte le même nom qu'une fonction importée : `predire = predire(...)`.
Dès qu'on assigne à un nom dans une fonction, Python le traite comme local **sur toute la
fonction**, y compris avant l'assignation.
→ Nommer le résultat autrement (`valeur = predire(...)`).

### `Multiple top-level modules discovered in a flat-layout`
Des `.py` traînent à la racine du projet : setuptools ne sait pas lequel est le paquet.
Arrive typiquement après un téléchargement fichier par fichier, qui aplatit l'arborescence.
→ Ranger le code dans un dossier de paquet avec `__init__.py`, les tests dans `tests/`.
→ Et déclarer explicitement dans `pyproject.toml` : `[tool.setuptools]` / `packages = ["co2"]`.

### `editable mode currently requires a setuptools-based build`
pip trop ancien (< 21.3) pour installer en mode éditable depuis un `pyproject.toml` seul.
→ Vérifier d'abord qu'on est dans le bon `.venv` (`which python`) : c'est souvent un
environnement parallèle créé automatiquement par VS Code quand une installation échoue.
→ Sinon `pip install --upgrade pip`.

---

## 2. Mise en place d'un projet

### 2.1 — Rituel de démarrage

Deux cas, selon que le projet est une **analyse** (des notebooks) ou un **paquet**
(du code destiné à être importé, testé, déployé).

**Projet d'analyse** — les dépendances se listent dans la commande.

```bash
mkdir -p ~/Projets/<projet>/data && cd ~/Projets/<projet>
python3 -m venv .venv
source .venv/bin/activate          # à refaire à CHAQUE nouveau terminal
pip install pandas numpy matplotlib scikit-learn jupyter ipykernel
code .                             # ouvrir LE PROJET, pas le dossier parent
```

**Projet à paquet** — l'arborescence et le `pyproject.toml` viennent d'abord ; un seul
`pip install` lit les dépendances déclarées dedans (voir 2.2 et 2.3).

```bash
mkdir -p ~/Projets/<projet>/<paquet> ~/Projets/<projet>/tests
cd ~/Projets/<projet>
python3 -m venv .venv && source .venv/bin/activate
touch <paquet>/__init__.py
# écrire pyproject.toml, puis :
pip install -e ".[dev]"            # installe le paquet + pandas, pytest, ruff…
pytest                             # doit déjà tourner (0 test collecté)
code .
```

Puis, dans les deux cas :

```bash
git init
cat > .gitignore << 'EOF'
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/
.DS_Store
*.egg-info/
EOF
git add .
git commit -m "Initialisation du projet"
gh repo create <projet> --public --source=. --push
```

- `git init` et `gh repo create` : **une seule fois** par projet.
- Projet d'analyse : `requirements.txt` écrit à la main, dépendances **directes**
  seulement. `pip freeze` capture tout l'environnement (100+ lignes) — réservé à la
  reproduction exacte.
- Projet à paquet : les dépendances vivent dans `pyproject.toml`, pas ailleurs.

### 2.2 — Structure d'un projet Python

**Le vocabulaire, qui explique la forme.**

- Un **module** est un fichier `.py`. Il contient ce qu'on veut : fonctions, constantes,
  classes. `preparation.py` est un module.
- Un **paquet** est un dossier contenant des modules et un `__init__.py`. Un seul module
  suffit : c'est la structure qui fait le paquet, pas le nombre.
- L'arborescence du disque **est** la syntaxe d'import. Le point sépare les niveaux comme
  un slash sépare les dossiers :

```python
from co2.preparation import preparer
#    ^^^  ^^^^^^^^^^^     ^^^^^^^^
#  paquet   module         objet
```

C'est vrai partout : `sklearn` est un paquet, `sklearn.linear_model` un module dedans,
`LinearRegression` une classe dans ce module. `math` est un simple module.

`__init__.py` est exécuté à l'import du paquet. Vide, il signale seulement « ceci est un
paquet ». On peut y écrire `from co2.preparation import preparer` pour offrir une façade
courte (`from co2 import preparer`) — inutile tant qu'il n'y a qu'un module.

Un module sait s'il est importé ou lancé :

```python
if __name__ == "__main__":      # s'exécute avec `python co2/preparation.py`
    ...                         # mais pas à l'import
```

**L'arborescence.**

```
co2-api/
├── co2/                 le paquet — ce que j'écris
│   ├── __init__.py      déclare co2/ comme paquet importable
│   └── preparation.py
├── tests/               le code qui vérifie le code
│   └── test_preparation.py
├── pyproject.toml       identité, dépendances, config des outils
├── README.md
└── .gitignore
```

- **Ce qui va dans Git** : ces six fichiers, rien d'autre.
- **Ce que les outils fabriquent**, et qui reste ignoré : `.venv/` (l'environnement,
  des centaines de Mo, dépendant de la machine), `__pycache__/` (bytecode),
  `.pytest_cache/` (dernier lancement, permet `pytest --lf`), `*.egg-info/` (carte
  d'identité écrite par `pip install -e` ; `top_level.txt` y contient le nom du paquet).
- **La vérification qui tranche** : `git ls-files` liste ce que Git suit réellement,
  par opposition à ce qu'affiche l'éditeur.
- Séparer `co2/` et `tests/` n'est pas cosmétique : à l'installation, seul le paquet part.
- `co2/` contient ce qui est **importé** : des définitions, sans effet de bord. `scripts/`
  contient ce qui est **exécuté** : arguments de ligne de commande, écriture de fichiers,
  affichage. Test : `import co2.modele` ne doit *rien faire*.
- `data/` et `models/` sont des intrants et des artefacts : hors Git dans un projet de
  service. Dans un projet d'analyse, versionner les données se défend (reproductibilité).
- Variante fréquente : `src/co2/` au lieu de `co2/`, pour forcer le travail sur la
  version installée plutôt que sur les fichiers du dossier courant.

### 2.3 — `pyproject.toml`

Le fichier d'identité du projet. Il a remplacé `setup.py`, `setup.cfg`, `pytest.ini`,
`.flake8`… Quatre rôles :

```toml
[project]                        # 1. qui je suis
name = "co2"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pandas", "numpy", "scikit-learn"]

[project.optional-dependencies]  # 2. ce qu'il faut pour développer
dev = ["pytest", "ruff"]

[build-system]                   # 3. quel outil sait me construire
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[tool.setuptools]                #    ne pas laisser deviner le paquet
packages = ["co2"]

[tool.pytest.ini_options]        # 4. la config des autres outils
testpaths = ["tests"]
[tool.ruff]
line-length = 100
```

Le 4e bloc explique la popularité du format : chaque outil vient lire sa section
`[tool.xxx]`. Une seule configuration pour tout le projet.

```bash
pip install -e ".[dev]"
```

`.` = dossier courant · `[dev]` = dépendances optionnelles · `-e` = mode **éditable**,
Python pointe vers mes fichiers au lieu d'en copier une version figée. C'est ce qui fait
marcher `from co2.preparation import preparer` sans bricoler `sys.path`.

### 2.4 — Messages de commit

Impératif présent : le message complète « ce commit… ». Sujet sous 50 caractères
(72 max), sans point final, majuscule à l'initiale. Le détail va dans le corps.

```bash
git commit -m "Extrait la preparation des donnees en module teste" -m "Corps : ce que le commit apporte, en phrases. C'est ce qu'on relira dans six mois."
```

- Le message se déduit du **diff**, jamais de ce qu'on croit avoir modifié :
  `git status` puis `git add`, puis `git diff --staged` pour voir le futur commit.
- `git diff --staged` vide = rien n'est indexé, pas « rien n'a changé ».
- Si le message ne vient pas, c'est souvent que le commit mélange deux choses.
- « Mise à jour » ne dit rien. Dire **laquelle**.
- Éviter les indices (`CO₂`) et les accents : mauvais rendu dans certains terminaux.
- Une convention par dépôt, et s'y tenir.
- Le `git diff` d'un notebook est illisible (JSON + base64). `git status` pour savoir
  quels fichiers ont bougé, la mémoire de la session pour savoir quoi.

---

## 3. Pandas — gestes courants

```python
# Charger et nettoyer
df = pd.read_csv("data/f.csv", sep=";", encoding="latin-1")
obj = df.select_dtypes(include="str").columns      # "object" avant pandas 3
df[obj] = df[obj].apply(lambda s: s.str.strip())

# Découvrir
df.shape ; df.head() ; df.info() ; df.describe() ; df.isna().sum()

# Sélectionner
df["col"]              # Series
df[["a", "b"]]         # DataFrame (le double crochet est une liste)
df.loc[2, "nom"]       # par étiquette
df.iloc[0:3]           # par position
# éviter l'indexation chaînée : df.iloc[2]["col"] → df.loc[2, "col"]

# Filtrer
df[(df["a"] > 30) & (df["b"] == "x")]
df[df["a"].isin(["x", "y"])]

# Agréger — TOUJOURS count à côté de mean
df.groupby("cat")["val"].agg(["mean", "count", "std"]).sort_values("mean")
df.groupby("cat")[["a", "b"]].count()   # croiser avec les manquantes

# Inspecter
df.nlargest(10, "col")
df.drop_duplicates(subset=[...])

# Encoder une catégorique en indicatrices
pd.get_dummies(df[num + cat], columns=cat, drop_first=True).astype(float)

# Découper une variable continue pour inspecter un motif
df.groupby(pd.qcut(pred, 10))["residu"].mean()   # déciles
df.groupby(pd.cut(df["x"], [0, 75, 150, 1000]))  # bornes choisies
```

**Pièges Pandas**

- `mean()` **ignore** les `nan` par défaut : le dénominateur change sans avertissement.
  `mean(skipna=False)` force la propagation ; `count()` donne le nombre de valeurs présentes.
- Une seule valeur manquante fait passer une colonne d'entiers en `float64`
  (`nan` est un flottant).
- `size()` compte les **lignes**, les agrégations numériques ne comptent que les
  valeurs **présentes** : comparer les deux révèle où sont les trous.
- Une moyenne sur 1 élément s'affiche comme les autres. D'où le `count`, toujours.
- `pd.get_dummies` : les lignes dont la colonne catégorielle vaut `nan` reçoivent des
  zéros partout — donc exactement le codage de la modalité de référence. Aucun
  avertissement, coefficient de référence biaisé.
  → `dummy_na=True`, ou filtrer les `nan` avant.
- `drop_duplicates` garde la **première** ligne rencontrée, pas la plus pertinente.
  Trier d'abord si le choix compte : `sort_values(...).drop_duplicates(...)`.
- Tout filtre en amont définit un **périmètre**, qui doit être annoncé.
  `df[df["energ"] == "ES"]` exclut aussi les hybrides, qui ont leur propre code.
  → `value_counts()` sur la colonne **avant** de filtrer, pour voir ce qu'on écarte.

---

## 4. NumPy — gestes courants

```python
a = np.array([...])              # 1D = vecteur
M = np.arange(12).reshape(3, 4)  # 2D = matrice

M.shape          # (3, 4)  ← le réflexe de débogage n°1
M[1]             # une ligne
M[:, 1]          # une colonne (le ':' est obligatoire)
M[-1]            # dernière ligne

A * B            # terme à terme
A @ B            # produit matriciel  ← ne PAS confondre
A.T              # transposée

C.mean(axis=0)                  # écrase les lignes → (4,)  s'aligne naturellement
C.mean(axis=1, keepdims=True)   # écrase les colonnes → (3,1) pour rebroadcaster

a[(a > 3) & (a < 9)]            # masque booléen
(a > 5).sum()                   # compter (True vaut 1)
resultat.size > 0               # un filtre peut ne rien renvoyer
np.where(a > 5, a, 0)           # remplacer
```

**À retenir**

- `axis` = **l'axe qui disparaît**. Une agrégation réduit d'une dimension.
- Broadcasting : formes alignées **à partir de la droite**, compatibles si égales
  ou si l'une vaut 1.
- `*` sur une **liste** Python répète la séquence ; sur un **array**, il multiplie.
  Même symbole, sens opposé.
- Un tableau vide (`shape (0,)`) ne lève pas d'erreur au filtrage — il explose plus loin.
- `np.full_like(a, valeur)` hérite du **dtype** de `a` : une moyenne placée dans un
  tableau d'entiers est tronquée sans avertissement.
  → `np.full(a.shape, valeur)` quand la valeur est flottante.

---

## 5. Machine learning — le socle

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

X = x.reshape(-1, 1)                      # sklearn veut du 2D
model = LinearRegression().fit(X, y)      # ← l'entraînement, c'est le .fit()
model.coef_, model.intercept_             # les paramètres appris
y_pred = model.predict(X)

mean_absolute_error(y, y_pred)   # erreur moyenne, dans l'unité de la cible
r2_score(y, y_pred)              # part de variation expliquée
```

Versions vectorisées des fonctions du cours :

```python
def compute_cost(x, y, w, b):
    m = x.shape[0]
    return (1 / (2 * m)) * np.sum((w * x + b - y) ** 2)

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    erreur = w * x + b - y            # terme commun aux deux dérivées
    return np.sum(erreur * x) / m, np.sum(erreur) / m
```

Normalisation, et retour aux unités d'origine :

```python
mu, sigma = x.mean(), x.std()
x_norm = (x - mu) / sigma
# après entraînement sur x_norm :
w_orig = w / sigma
b_orig = b - w * mu / sigma
```

**À retenir**

- Vocabulaire : **modèle** = `f`, **algorithme d'apprentissage** = descente de gradient,
  **entraînement** = le `.fit()`. Les paramètres (`w`, `b`) sont appris ;
  les hyperparamètres (α) sont choisis.
- Le `1/2` du coût n'a aucun sens statistique : il simplifie la dérivée. `J × 2` = MSE.
- Toujours mettre à jour `w` et `b` **simultanément** (calculer les deux gradients avant).
- Diagnostic de base : tracer le coût en fonction des itérations. Il doit décroître.
- Conserver `mu` et `sigma` — les paramètres appris portent sur la variable transformée.
- Ne jamais extrapoler hors de la plage d'entraînement (d'où un `b` sans sens physique).

**Régression multiple**

```python
# Évaluer hors échantillon dès que le nombre de paramètres varie
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
m = LinearRegression().fit(Xtr, ytr)
r2_score(yte, m.predict(Xte))

# Référence naïve : prédire la moyenne pour tout le monde
mean_absolute_error(yte, np.full(yte.shape, ytr.mean()))
```

- Ajouter une variable ne peut **jamais** faire baisser le R² d'entraînement.
  Dès que le nombre de paramètres varie d'un modèle à l'autre, il cesse d'être un
  critère de comparaison → jeu de test, ou R² ajusté.
- Toujours comparer à une **référence naïve**. Une MAE seule ne veut rien dire ;
  un rapport à la référence, si.
- `R²(A) + R²(B) ≠ R²(A, B)` dès que A et B sont corrélées : l'information est
  partagée, pas additive. Le bon indicateur est le **gain incrémental**, pas le R²
  de chaque variable prise seule.
- Hiérarchie polynomiale : ne pas retenir un terme d'ordre supérieur sans son terme
  d'ordre inférieur. Sans terme linéaire, la parabole a son sommet forcé en zéro.
- Encoder une catégorie : le one-hot brut laisse chaque modalité libre ; décomposer
  (`"A 8"` → type + nombre de rapports) impose linéarité **et** additivité. Moins de
  colonnes, moins de gain — l'arbitrage se mesure, il ne se devine pas.

---

## 6. Hygiène de notebook

- **Restart and Run All** avant chaque commit. Sans ça, les états fantômes passent en ligne.
- Les numéros `In[ ]` doivent être dans l'ordre : sinon le notebook n'a pas été
  exécuté de bout en bout.
- Une variable réutilisée d'une cellule à l'autre (`w`, `b`, `df`…) est la source
  n°1 de résultats faux sans erreur.
- Chaque chiffre écrit dans une cellule Markdown doit avoir été **mesuré**, pas estimé.
- Une cellule = une idée. Seule la dernière expression s'affiche
  (sinon `display()` ou séparer).
- Markdown avant le code (ce que je cherche), Markdown après la sortie (ce que j'y lis).

---

## 7. Algorithmique — repères

| Complexité | Comportement | Exemple |
|---|---|---|
| `O(1)` | constant | accès à une clé de dictionnaire |
| `O(log n)` | divisé par 2 à chaque étape | recherche dichotomique |
| `O(n)` | proportionnel | parcours d'une liste |
| `O(n log n)` | optimal pour un tri par comparaison | tri fusion |
| `O(n²)` | explose vite | deux boucles imbriquées |

- **Early exit** : sortir dès que la réponse est acquise (`return False` au premier
  contre-exemple), plutôt qu'un drapeau mis à jour jusqu'au bout.
- **Exploiter ce qu'on sait déjà** : liste triée → dichotomie ; deux moitiés triées →
  fusion en `O(n)` et non `sorted()` en `O(n log n)`.
- **Deux pointeurs** : comparer par les deux bouts en convergeant vers le centre.
- **Diviser pour régner** : couper en deux, résoudre chaque moitié (récursivement),
  recombiner. Le cas d'arrêt fait le travail ; le tri se fait à la **remontée**.
- Un test d'appartenance `in` sur une **liste** est `O(n)` ; sur un **set**, quasi `O(1)`.
- Deux algorithmes de même complexité ne sont pas équivalents pour autant
  (nombre de comparaisons vs nombre d'écritures).

---

## 8. Méthode d'analyse

- Devant un `nan` : demander **pourquoi il est absent**, pas comment le combler.
  Absence structurelle (liée à une catégorie) ≠ absence accidentelle.
  Un `dropna()` global peut supprimer une catégorie entière.
- Croiser les valeurs manquantes avec une variable catégorielle révèle le motif.
- Une valeur aberrante se **vérifie** avant d'être écartée : recouper avec une autre
  colonne pour juger de sa plausibilité.
- Corrélation = alignement (sans unité, entre −1 et 1).
  Pente = de combien y varie par unité de x (avec unité).
  Ni l'une ni l'autre n'est une cause.
- Une corrélation faible ne signifie pas absence de relation : le coefficient ne
  mesure que la **linéarité**. Toujours regarder le nuage de points.
- Un indicateur calculé sur deux périmètres différents n'est pas comparable
  (mélanger deux sous-populations brouille une relation ; isoler une catégorie peut la révéler).
- Les **résidus** disent *où* le modèle se trompe — le graphique le plus informatif.
  Erreurs qui s'évasent ou traînée asymétrique = variable manquante.
- Le choix de la métrique décide de la conclusion. Toujours savoir ce qu'on mesure.
- Nommer les limites de son analyse : périmètre, conditions de mesure, exclusions,
  effets non contrôlés.
- **Fuite de données** (*data leakage*) : une variable qui contient l'information de
  la cible d'une façon indisponible au moment de prédire. Symptôme : un R² anormalement
  élevé. Test : cette variable serait-elle connue *avant* la cible, en situation réelle ?
- Une fuite **non linéaire** ne se voit pas dans une matrice de corrélation
  (`puiss_admin` : 0,708 seulement, alors que la formule officielle la reproduit à
  99,5 %). Chercher la relation, pas seulement le coefficient.
- Deux autres tests de fuite qui fonctionnent : le rapport `cible / variable` est-il
  quasi constant ? Le coefficient appris coïncide-t-il avec une constante physique
  connue ?
- **Une ligne n'est pas forcément une observation.** Vérifier l'unité du fichier avant
  tout calcul : un modèle déclaré en 18 variantes pèse 18 fois plus lourd dans la somme
  des carrés. Dédoublonner sur des clés métier explicites.
  - **Dédoublonner AVANT de découper.** Si le même objet figure en plusieurs lignes,
  un `train_test_split` en place des copies des deux côtés : le modèle est évalué
  sur des observations qu'il a déjà vues. C'est une fuite, et elle gonfle le score
  de test sans rien signaler.
- Un coefficient au signe **physiquement absurde** n'est pas forcément de la
  colinéarité. Vérifier dans l'ordre : corrélation entre variables, R² de la variable
  seule, gain incrémental. Un gain nul → le coefficient ajuste du bruit, il ne
  s'interprète pas.
- **Se méfier des résultats mécaniques.** Le résidu moyen par modalité d'une variable
  présente dans le modèle est nul par construction (moindres carrés) ; une colonne
  constante donne le même zéro. Aucun des deux ne démontre quoi que ce soit.
- Un R² de test **supérieur** au R² d'entraînement signale un tirage favorable, pas un
  bon modèle. Le classement des modèles reste valide s'ils partagent le découpage ;
  les valeurs absolues, non. → validation croisée.
- Détecter une courbure résiduelle sans se fier à l'œil :
  `df.groupby(pd.qcut(pred, 10))["residu"].mean()`. Un motif en U = non-linéarité
  restante ; des moyennes qui oscillent autour de zéro = rien à voir.
- Un modèle ne peut pas être meilleur que ses variables. Quand deux objets que tout
  sépare en réalité sont identiques dans le tableau, l'erreur est irréductible :
  c'est une limite de données, pas d'algorithme.

---

## 9. Du notebook au module

### 9.1 — Ce qui change

Le passage du `.ipynb` au `.py` n'est pas un copier-coller. Trois habitudes de notebook
deviennent des défauts dans un module.

| Dans un notebook | Dans un module | Pourquoi |
|---|---|---|
| la fonction lit `df` global | `df` passe en argument | sinon intestable, et casse au premier import |
| `print(...)` | `return ...` | un print est invisible depuis un test |
| on modifie le DataFrame en place | `.copy()` d'abord | un effet de bord en production est un bug qui ne se reproduit pas |

### 9.2 — Tester avec pytest

```python
# tests/test_preparation.py
import pytest
from co2.preparation import preparer

@pytest.fixture
def brut():
    """Jeu synthétique : quelques lignes choisies pour être lisibles."""
    return pd.DataFrame({...})

def test_deduplique_les_variantes(brut):
    assert len(preparer(brut, "ES")) == 3
```

- Tester sur des **données inventées**, pas sur le vrai fichier : quelques millisecondes,
  des cas choisis, et aucune dépendance à un CSV.
- Une `@pytest.fixture` est un jeu de données préparé une fois et réinjecté dans chaque
  test qui le demande en argument.
- `pytest.approx(0.25)` pour comparer des flottants.
- `with pytest.raises(ValueError, match="..."):` pour vérifier qu'un cas invalide échoue
  **explicitement** plutôt que de renvoyer du vide.
- Vérifier aussi l'absence d'effet de bord : `pd.testing.assert_frame_equal(entree, copie)`.

```bash
pytest              # tout
pytest -q           # sortie courte
pytest --lf         # seulement les tests qui ont échoué la dernière fois
pytest -k dedup     # ceux dont le nom contient "dedup"
```

**À retenir**

- Un test qui échoue peut désigner une **ambiguïté dans la fonction**, pas une erreur
  dans le test. Lire l'écart avant de corriger le chiffre attendu.
- Écrire le test **d'abord**, le voir échouer, puis corriger la fonction. C'est l'ordre
  qui garantit que le test teste vraiment quelque chose.
- Une fonction qui fait deux choses (filtrer *et* dédoublonner) donne un test ambigu.
  Le test force à séparer — c'est son second bénéfice, après la vérification.
- Faire échouer un test volontairement de temps en temps : savoir lire une sortie pytest
  est une compétence à part entière.

**Lire une sortie pytest**

```
tests/test_modele.py .......                        [ 50%]
tests/test_preparation.py ..F....                   [100%]
```

- Un caractère par test : `.` réussi · `F` échec d'assertion · `E` erreur (souvent dans
  la fixture) · `s` sauté. La position du `F` désigne lequel.
- Le pourcentage est une **progression**, pas un score : part des tests collectés déjà
  exécutés. Utile quand un test bloque — il dit où l'exécution s'est arrêtée.
- `>` marque la ligne qui a échoué, les `E` l'expliquent. Pytest réécrit les assertions
  pour afficher les valeurs réelles (`assert 3 == 2`) — un `assert` Python nu ne dirait rien.
- `configfile:` et `testpaths:` en tête : à vérifier en premier si un test n'est pas collecté.
- Les fichiers sont collectés dans l'**ordre alphabétique**, pas celui de leur écriture.
- Le second argument d'une assertion est un message qui dit ce que le test croyait
  vérifier : `assert len(x) == 2, "les deux finitions doivent survivre"`. Sur les
  assertions non évidentes seulement — ailleurs, le nom du test suffit.

**Outils pytest à connaître**

- `tmp_path` : dossier temporaire fourni par pytest, nettoyé après le test. Pour tout ce
  qui écrit sur disque, au lieu de polluer le projet.
- `with pytest.warns(UserWarning, match="..."):` : transforme un avertissement attendu en
  garantie vérifiée, au lieu de le laisser traîner en bas de la sortie.

---

### 9.3 — Du modèle à l'artefact

**Le cycle, valable partout** : construire → entraîner → **évaluer** → sauvegarder →
charger → prédire. Ne jamais sauter l'évaluation : faire retourner à `entrainer()` le
modèle *et* ses métriques empêche un modèle non mesuré de circuler.

**Tout mettre dans un `Pipeline`.** Le problème : au moment de prédire, les données
doivent subir exactement les mêmes transformations qu'à l'entraînement. Un `drop_first`
oublié ou des colonnes dans un autre ordre produisent des prédictions fausses *sans lever
d'erreur*. Ça porte un nom : **train/serve skew**.

```python
preparation = ColumnTransformer([
    ("masse",     "passthrough",                           ["masse_ordma_min"]),
    ("puissance", PolynomialFeatures(2, include_bias=False), ["puiss_max"]),
    ("boite",     OneHotEncoder(handle_unknown="ignore",
                                drop="first", sparse_output=False), ["typ_boite_nb_rapp"]),
])
modele = Pipeline([("preparation", preparation), ("regression", LinearRegression())])
```

Le fichier sérialisé contient alors les transformations **et** les coefficients : il
devient impossible de prédire avec un encodage différent.

- `PolynomialFeatures(2)` sur une colonne = la colonne et son carré. C'est le
  `d["x²"] = d["x"]**2` du notebook, sous une forme que le pipeline sait rejouer.
- `OneHotEncoder` = `pd.get_dummies`, `drop="first"` = `drop_first=True`.
- `ColumnTransformer` applique chaque traitement à la bonne colonne ; `Pipeline` enchaîne
  préparation puis modèle. Un seul `.fit()`, un seul `.predict()`.
- `handle_unknown="ignore"` : une modalité jamais vue devient des zéros au lieu de lever
  une exception. Indispensable pour une API — arbitrage explicite, à documenter.
- Inspecter ce que ça produit : `modele.named_steps["preparation"].get_feature_names_out()`

**Sérialiser.** `joblib.dump` / `joblib.load` — même principe que `pickle`, optimisé pour
les gros tableaux NumPy. L'extension `.joblib` est une convention. Le `.joblib` est un
**artefact** : régénérable, donc hors Git (`models/` dans `.gitignore`).

**Entraîner dans un script, jamais dans l'API.** Un service qui entraîne au démarrage rend
chaque redéploiement imprévisible. Le script (`scripts/entrainer.py` + `argparse`) est la
**trace de comment l'artefact a été produit** : quelle motorisation, quel fichier, quel
`random_state`. Un `.joblib` sorti d'un REPL est un artefact sans provenance.

**Valider à la frontière.** Un modèle linéaire n'a pas de notion de plausibilité : masse
négative → CO₂ négatif, sans avertissement. La validation se fait **là où les données
entrent dans le système** (l'API), pas dans la fonction de prédiction, qui est une
bibliothèque et peut légitimement extrapoler.

**À retenir**

- Un test dit ce qui **doit** se passer, pas ce qui se passe. Écrire un test qui constate
  un comportement qu'on juge mauvais le transforme en contrat.
- Le REPL sert à **découvrir** un comportement, le test à le **verrouiller** une fois décidé.
- Le test qui protège vraiment : sauvegarder, recharger, vérifier que les prédictions sont
  identiques (`np.testing.assert_allclose`).
