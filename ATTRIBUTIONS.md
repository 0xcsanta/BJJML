# Attributions

Inventaire de toute source tierce utilisée par ce projet : d'où elle vient, sous quelle licence,
et ce qu'on en fait exactement.

**Aucune de ces données n'est présente dans ce dépôt.** Chacune est téléchargée à la demande par
un script de `src/jjbml/fetch/`, vers `data/`, qui est intégralement gitignoré. Chaque
téléchargement écrit un `data/raw/<source>/SOURCE.json` contenant l'URL d'origine, la date, la
taille et l'empreinte SHA-256 de chaque fichier — de quoi vérifier à tout moment d'où vient
chaque octet présent sur le disque.

---

## ViCoS Brazilian Jiu-Jitsu Positions Dataset

- **Origine** : https://vicos.si/resources/jiujitsu/ — ViCoS Lab, Université de Ljubljana
- **Licence** : [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Récupéré par** : `python -m jjbml.fetch.vicos`
- **Contenu** : 120 279 images labellisées issues de 6 séquences de sparring no-gi, 10 positions
  réparties en 18 classes, keypoints 2D au format MS-COCO (17 articulations, `[x, y, confiance]`)
  pour les deux athlètes.
- **Usage dans le projet** : amorçage de l'étage A (perception). Les keypoints fournis permettent
  d'entraîner un premier classifieur de position sans résoudre la vision.

**Citation requise par la licence :**

> Hudovernik, V. et Škočaj, D. — *Video-Based Detection of Combat Positions and Automatic Scoring
> in Jiu-jitsu.* MMSports '22 : 5th International ACM Workshop on Multimedia Content Analysis in
> Sports, Lisbonne, 14 octobre 2022, p. 55-63.

**Obligations respectées :**
- *BY* — citation ci-dessus, reprise dans le README et vérifiée par le hook pre-commit.
- *NC* — le projet est non commercial.
- *SA* — toute donnée dérivée redistribuée portera la même licence. Les poids de modèles
  entraînés sur ce jeu de données, s'ils sont publiés, le seront sous CC BY-NC-SA 4.0.

---

## GrappleMap

- **Origine** : https://github.com/Eelis/GrappleMap — Eelis van der Weegen
- **Licence** : domaine public (« the GrappleMap code and data is released into the public domain »)
- **Récupéré par** : `python -m jjbml.fetch.grapplemap`
- **Contenu** : `GrappleMap.txt`, graphe orienté de positions et transitions construit à la main
  à partir de sources instructionales. 39 098 lignes, 919 entrées taguées, 161 tags distincts,
  coordonnées d'articulations encodées.
- **Usage dans le projet** : ossature du graphe (étage B) et vocabulaire fin des grips
  (`top_underhook`, `crossface`, `collar_tie`…), beaucoup plus précis que les 10 classes de ViCoS.
- **Note** : le format est ad-hoc et non documenté, l'auteur prévient qu'il peut changer.

Aucune obligation d'attribution n'est attachée au domaine public. La mention est faite par
honnêteté.

---

## bjjgraph

- **Origine** : https://github.com/diogoseca/bjjgraph — Diogo Seca
- **Licence** : [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)
  (« source-available, not open source »)
- **Récupéré par** : `python -m jjbml.fetch.bjjgraph` (actif `graph.json.gz` de la dernière release)
- **Contenu** : 133 positions, 1 025 transitions, 353 soumissions, avec des valeurs distinctes
  pour le gi et le no-gi.
- **Usage dans le projet** : **la structure du graphe uniquement.**

**Avertissement important.** Les probabilités publiées par cette source sont, selon son propre
README, des *authored estimates* : écrites à la main, régénérables par enrichissement IA, et en
attente d'une revue par un panel de ceintures noires qui n'a pas eu lieu. Ce ne sont pas des
mesures. Ce projet ne les utilise jamais comme vérité terrain ; elles servent uniquement de point
de comparaison pour les valeurs mesurées sur des matchs réels (cf. PLAN.md, phase 6).

**Obligations respectées :** usage non commercial ; aucun scraping du site (on utilise les
artefacts publiés par le dépôt, prévus pour cet usage).

---

## Vidéos de compétition (phase 4)

- **Origine** : plateformes publiques de diffusion (ADCC, Who's Number One, Polaris…)
- **Statut** : œuvres protégées par le droit d'auteur de leurs ayants droit.
- **Usage dans le projet** : annotation de timelines de positions.

**Règle stricte.** Aucune vidéo ni aucune frame extraite n'entre dans ce dépôt, sous aucune
forme. Les annotations produites ne contiennent que `video_id + timestamp + label` : des faits
relatifs à un match, qui ne sont pas protégeables, et non une reproduction de l'œuvre. Le hook
pre-commit refuse tout média indexé.

---

## PolyForm Noncommercial License 1.0.0

Le fichier [LICENSE](LICENSE) reproduit le texte officiel de la licence, récupéré depuis
https://github.com/polyformproject/polyform-licenses (branche `1.0.0`).
