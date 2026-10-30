# jjb-ML

Un moteur de jiu-jitsu brésilien **no-gi** construit comme un *livre d'ouvertures aux échecs* :
on lui décrit une situation entre deux combattants, il propose les meilleures options pour un
joueur donné et les enchaînements probables selon la réaction de l'adversaire.

Projet d'apprentissage du machine learning, **non commercial**. Le plan complet est dans
[PLAN.md](PLAN.md).

## Ce que c'est, et ce que ce n'est pas

Ce n'est **pas** un AlphaZero du grappling : il n'existe aucun simulateur physique du jiu-jitsu,
donc pas de self-play possible. C'est un moteur **statistique**, qui apprend ce qui marche
réellement chez les compétiteurs de haut niveau, et qui le mesure au lieu de le supposer.

L'ossature du graphe des positions existe déjà dans des bases publiques. Ce que personne n'a
jamais fait, c'est **peser les arêtes** avec des données réelles. C'est l'objet du projet.

## Architecture

| Étage | Rôle |
|---|---|
| **A — Perception** | vidéo → timeline d'états (pose, tracking, classification temporelle) |
| **B — Le graphe** | positions et transitions, récupérées de sources existantes et unifiées |
| **C — Le moteur** | poids mesurés sur les arêtes + recherche dans l'arbre → recommandations |

L'utilisateur **déclare** la position qu'il subit et qui il est ; la perception propose et il
corrige. Le moteur rend donc un service utile même quand la vision est imparfaite, et chaque
correction devient une donnée annotée.

## Démarrage

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -e .
python -m jjbml.fetch.all     # telecharge les sources dans data/ (gitignore)
```

`python -m jjbml.fetch.vicos --images` télécharge en plus les 9,35 Go d'images ViCoS. Elles ne
sont **pas** nécessaires avant la phase 4 : les keypoints sont déjà dans `annotations.json`
(199 Mo), ce qui suffit à entraîner le premier modèle.

## Sources et licences

Ce dépôt ne contient **aucune donnée tierce**. Chaque source est téléchargée par un script dans
`src/jjbml/fetch/`, vers `data/`, qui est intégralement gitignoré. Le détail complet est dans
[ATTRIBUTIONS.md](ATTRIBUTIONS.md).

| Source | Licence | Usage |
|---|---|---|
| [ViCoS BJJ Positions Dataset](https://vicos.si/resources/jiujitsu/) | CC BY-NC-SA 4.0 | Amorçage de la perception |
| [GrappleMap](https://github.com/Eelis/GrappleMap) | Domaine public | Graphe et vocabulaire des grips |
| [bjjgraph](https://github.com/diogoseca/bjjgraph) | PolyForm Noncommercial 1.0.0 | Ontologie — **structure uniquement** |

### Citation requise

Le ViCoS BJJ Positions Dataset est utilisé conformément à sa licence CC BY-NC-SA 4.0 :

> **Hudovernik, V. et Škočaj, D.** — *Video-Based Detection of Combat Positions and Automatic
> Scoring in Jiu-jitsu.* MMSports '22 : 5th International ACM Workshop on Multimedia Content
> Analysis in Sports, Lisbonne, 14 octobre 2022, p. 55-63.

### Avertissement sur bjjgraph

Les probabilités publiées par bjjgraph sont des *authored estimates* — écrites à la main et non
mesurées. Ce projet n'en reprend que **la structure du graphe**. Les probabilités servent
uniquement de point de comparaison pour les valeurs mesurées, jamais de vérité terrain.

## Statut non commercial

Les licences amont (CC BY-NC-SA 4.0 et PolyForm Noncommercial 1.0.0) l'imposent, et le projet
l'assume. Le code de ce dépôt est publié sous **PolyForm Noncommercial 1.0.0** — voir
[LICENSE](LICENSE).

## Conformité

Les règles sont dans [PLAN.md](PLAN.md), section 3, et elles sont appliquées automatiquement par
un hook qui refuse tout commit non conforme :

```bash
git config core.hooksPath tools/hooks
```

Le hook refuse un fichier sous `data/`, un média ou une archive, un fichier de plus de 1 Mo, la
disparition de la citation ViCoS du README, ou l'absence d'`ATTRIBUTIONS.md`.
