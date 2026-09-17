# ml-notes

Carnet technique personnel, alimenté au fil de mon parcours vers l'ingénierie IA.

Ce dépôt est un **carnet vivant**, pas un cours : il rassemble ce que j'ai
réellement rencontré, avec le diagnostic que j'ai appliqué. Il est incomplet
par nature et s'enrichit au fur et à mesure.

## Contenu

- **[`MEMO.md`](MEMO.md)** — le carnet de terrain. Erreurs indexées par leur message
  exact (§1), mise en place d'un projet — rituel de démarrage, structure de paquet,
  `pyproject.toml`, messages de commit (§2), gestes Pandas et NumPy (§3-4),
  socle ML (§5), hygiène de notebook (§6), complexité et réflexes algorithmiques (§7),
  méthode d'analyse (§8), passage du notebook au module et tests `pytest` (§9).
  Fait pour être cherché (`Ctrl+F`), pas lu d'un bout à l'autre — le sommaire en tête
  de fichier renvoie vers chaque section.
- **[`memo-machine-learning.pdf`](memo-machine-learning.pdf)** — les principes, en slides. Le « pourquoi »
  plutôt que le « quoi taper ». À relire d'un bloc.
  ([source `memo-machine-learning.pptx`](memo-machine-learning.pptx) pour l'édition)

## Règle d'alimentation

Dès qu'un problème me coûte plus de dix minutes, il gagne trois lignes dans
`MEMO.md`, au moment où il arrive. Même chose pour une décision tranchée dans un
projet : sans ça, le carnet diverge du code.

## Projets associés

- [analyse-co2-vehicules](https://github.com/aissatougueye06/analyse-co2-vehicules) — analyse exploratoire (pandas)
- [ml-co2-prediction](https://github.com/aissatougueye06/ml-co2-prediction) — application de la ML Specialization
