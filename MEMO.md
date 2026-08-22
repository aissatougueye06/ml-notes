# MEMO — carnet technique

Carnet personnel, alimenté au fil des projets et du cours
[Machine Learning Specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/).

**Ce qui va ici** : les lignes que je retape sans arrêt, et les messages d'erreur avec
leur cause. **Ce qui ne va pas ici** : les principes et le « pourquoi » — ils sont dans
`memo-machine-learning.pptx`, à relire d'un bloc.

**Règle d'alimentation** : dès qu'un problème m'a coûté plus de dix minutes, il gagne
trois lignes ici, au moment où il arrive.

---

## 1. Messages d'erreur → cause → correctif

Indexés par le **texte exact** du message : c'est ce que j'aurai sous les yeux, pas la cause.

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

### `nan` inattendu dans un résultat, sans erreur
Souvent une variable réutilisée d'une cellule précédente (état fantôme de notebook).
→ *Kernel → Restart and Run All*. Vérifier que les numéros `In[ ]` sont dans l'ordre.

### `cannot reshape array of size 12 into shape (5,3)`
`reshape` ne crée ni ne détruit de données : lignes × colonnes doit égaler le total.

---

## 2. Mise en place d'un projet

Rituel complet, du dossier vide au dépôt en ligne.

```bash
mkdir -p ~/Projets/<projet>/data && cd ~/Projets/<projet>
python3 -m venv .venv
source .venv/bin/activate          # à refaire à CHAQUE nouveau terminal
pip install pandas numpy matplotlib scikit-learn jupyter ipykernel
code .                             # ouvrir LE PROJET, pas le dossier parent
```

```bash
git init
cat > .gitignore << 'EOF'
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
EOF
git add .
git commit -m "Initialisation du projet"
gh repo create <projet> --public --source=. --push
```

Ensuite, à chaque étape terminée :

```bash
git add . && git commit -m "<ce que ce commit apporte>" && git push
```

- `git init` et `gh repo create` : **une seule fois** par projet.
- `git status` : voir ce qui a changé et n'est pas encore sauvegardé.
- `requirements.txt` : écrire les dépendances **directes** à la main.
  `pip freeze` capture tout l'environnement (100+ lignes) — réservé à la reproduction exacte.

---

## 3. Pandas — gestes courants

```python
# Charger et nettoyer
df = pd.read_csv("data/f.csv", sep=";", encoding="latin-1")
obj = df.select_dtypes(include="object").columns   # ou "str" selon la version
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
```

**Pièges Pandas**

- `mean()` **ignore** les `nan` par défaut : le dénominateur change sans avertissement.
  `mean(skipna=False)` force la propagation ; `count()` donne le nombre de valeurs présentes.
- Une seule valeur manquante fait passer une colonne d'entiers en `float64`
  (`nan` est un flottant).
- `size()` compte les **lignes**, les agrégations numériques ne comptent que les
  valeurs **présentes** : comparer les deux révèle où sont les trous.
- Une moyenne sur 1 élément s'affiche comme les autres. D'où le `count`, toujours.

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
