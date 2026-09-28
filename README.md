# ml-notes

Je viens de la gestion de projet IT et j'apprends l'ingénierie ML. Ce dépôt est le
carnet que je tiens en chemin.

Ce n'est pas un cours, c'est un journal de terrain : ce que j'ai rencontré en
construisant des projets, et le diagnostic que j'ai appliqué. Les erreurs y sont
indexées par leur **message exact** — celui qu'on a sous les yeux, pas la cause qu'on
découvre après.

**Pile couverte** : Python (packaging, `pyproject.toml`, environnements virtuels),
pandas, NumPy, scikit-learn, pytest, Git.

## Projets

- **[co2-api](https://github.com/aissatougueye06/co2-api)** — prédiction des émissions
  de CO2 des véhicules (données ADEME). Paquet installable, pipeline scikit-learn
  sérialisé, tests pytest.
- **[ml-co2-prediction](https://github.com/aissatougueye06/ml-co2-prediction)** — mise
  en application de la
  [ML Specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/).
- **[analyse-co2-vehicules](https://github.com/aissatougueye06/analyse-co2-vehicules)** —
  analyse exploratoire (pandas).

## Contenu

- **[`MEMO.md`](MEMO.md)** — le carnet de terrain. Erreurs indexées par message, mise en
  place de projet, gestes pandas et NumPy, socle ML, hygiène de notebook, méthode
  d'analyse, passage du notebook au module. Fait pour être cherché (`Ctrl+F`), pas lu
  d'un bout à l'autre — le sommaire est en tête du fichier.
- **[`GLOSSAIRE.md`](GLOSSAIRE.md)** — les termes rencontrés, définis en trois lignes.
  Répond à « ça veut dire quoi, déjà ? ».
- **[`memo-machine-learning.pdf`](memo-machine-learning.pdf)** — les principes, en
  slides : le « pourquoi » plutôt que le « quoi taper ». À relire d'un bloc.
  ([source `.pptx`](memo-machine-learning.pptx) pour l'édition)

## Règle d'alimentation

Dès qu'un problème me coûte plus de dix minutes, il gagne trois lignes dans `MEMO.md`,
au moment où il arrive — comme toute décision tranchée dans un projet. Sans ça, le
carnet diverge du code.
