# Phase 2 — premier classifieur de position

Entrée : 104 caractéristiques (17 articulations COCO × (x, y, confiance) × 2 athlètes,
plus 2 indicateurs de présence). Sortie : l'une des 18 classes ViCoS.
Aucune image n'est utilisée.

---

## Le protocole a dû être changé en cours de route

Le plan prévoyait un **leave-one-roll-out** : tenir un combat entier à l'écart.
C'est impossible sur ce jeu de données, et la raison est structurelle.

Chaque roll de ViCoS est consacré à un petit groupe de positions :

| roll | frames | classes | contenu |
|---|---|---|---|
| 0 | 25 812 | 5 | gardes ouverte et fermée |
| 1 | 19 090 | 4 | 50-50 et demi-garde |
| 2 | 22 772 | 5 | side control et mount |
| 3 | 16 241 | 3 | tortue |
| 4 | 23 145 | 9 | debout et takedowns |
| 5 | 13 219 | 2 | contrôle du dos |

`back1` et `back2` n'existent que dans le roll 5, la tortue que dans le roll 3,
le mount que dans le roll 2. Écarter un roll retire donc ses classes de
l'entraînement : le modèle doit prédire des classes jamais vues. Mesuré,
**11 classes sur 18** se retrouvent dans ce cas, et le score s'effondre à 0,078
sans rien dire du modèle.

Ce n'est donc pas du sparring libre comme l'annonce la page du dataset, mais
des séances enregistrées position par position.

### Le protocole retenu

**Leave-one-view-out** : trois plis, chacun testé sur une caméra absente de
l'entraînement. Toutes les classes restent présentes des deux côtés.

**Ce qu'il mesure :** la généralisation à un nouvel angle de prise de vue.
**Ce qu'il ne mesure pas :** la généralisation à un nouveau combat. Les mêmes
athlètes, la même salle et les mêmes instants sont dans l'entraînement, vus
d'ailleurs. Tout chiffre ci-dessous est donc un **majorant optimiste**.

---

## Résultats

Accuracy équilibrée (rappel moyen par classe), 18 classes, leave-one-view-out :

| modèle | équilibrée | brute | 10 positions |
|---|---|---|---|
| classe majoritaire | 0,056 | 0,139 | 0,100 |
| régression logistique | 0,634 | 0,692 | 0,670 |
| MLP 256-128 | 0,731 | 0,757 | 0,752 |
| **MLP 256-128 + échange d'athlètes** | **0,746** | **0,773** | **0,771** |

La classe majoritaire tombe à 0,056, soit exactement 1/18 : c'est le hasard sur
cette métrique, et c'est le point de départ honnête.

Le réseau gagne **11 points** sur la régression logistique : la relation entre
géométrie des poses et position n'est pas linéaire, ce qui justifie le réseau.
L'augmentation par échange des deux athlètes ajoute 1,5 point et, surtout, force
le modèle à lire la géométrie plutôt que l'ordre dans lequel le détecteur a
rangé les deux personnes.

### Ce qu'aurait donné un split malhonnête

| protocole | équilibrée |
|---|---|
| split aléatoire par image | 0,875 |
| leave-one-view-out | 0,746 |

**13 points de fuite.** À 30 images par seconde, la frame 1047 d'entraînement et
la 1048 de test sont la même image — et la même scène existe en plus sous deux
autres angles. C'est le chiffre qu'on aurait annoncé sans y regarder.

---

## Où le modèle se trompe

Matrice de confusion sur les 10 positions, en % de chaque ligne :

```
       stand 5050_ taked open_ close half_ side_ mount  back turtl
standing    93*    0     6     0     0     0     0     0     0     0
5050_guar     0    79*    0     6     3     3     4     1     3     0
takedown    11     0    76*    3     1     1     0     0     4     3
open_guar     1     4     2    86*    1     4     1     1     1     0
closed_gu     0     2     1     2    74*    8     4     3     2     3
half_guar     0     2     0     5     5    65*    7    10     2     4
side_cont     0     1     0     1     3     8    67*   13     3     3
mount         0     1     0     0     2    11    14    67*    2     2
 back         0     2     4     1     1     2     5     2    81*    3
turtle        0     0     3     1     1     3     4     3     2    82*
```

Les erreurs sont **anatomiquement cohérentes**, ce qui est plutôt rassurant :
le bloc mount / side control / demi-garde se confond avec lui-même (mount →
side control 14 %, side control → mount 13 %, demi-garde → mount 10 %). C'est
exactement le continuum du passage de garde, où la frontière est floue même
pour un œil humain. Le modèle fait les bonnes erreurs.

Les classes les plus faibles sont `half_guard1` (0,641) et `takedown2` (0,617) ;
la plus solide est `standing` (0,927).

---

## Le vrai plafond n'est pas le modèle

| sous-ensemble | n | équilibrée |
|---|---|---|
| les deux athlètes détectés | 76 343 | **0,877** |
| un seul athlète détecté | 43 936 | **0,516** |

Un tiers du jeu de données n'a qu'une seule pose, parce que le détecteur perd
ou fusionne un corps quand les deux athlètes sont emmêlés — et ça frappe les
positions les plus serrées (`turtle1` perd une pose 3 990 fois).

**L'écart est de 36 points.** Quand la perception fait son travail, un simple
MLP atteint 0,877. Le facteur limitant de cette phase n'est donc pas
l'architecture du classifieur : c'est l'estimation de pose sur des corps
enchevêtrés.

Conséquence pour la suite : ajouter des couches ne servirait à rien. L'effort
doit aller à la perception (phase 3) et à des données réelles (phase 4).

---

## Ce qui est acquis, et ce qui ne l'est pas

Acquis : un classifieur qui bat largement les baselines, un protocole dont on
sait exactement ce qu'il mesure, et une mesure chiffrée de la fuite qu'un split
naïf aurait produite.

Pas acquis : **aucune preuve que ce modèle fonctionne sur un autre combat.**
Ce jeu de données ne permet structurellement pas de le vérifier. La phase 4,
annoter de la compétition réelle, n'est donc plus une amélioration souhaitable
mais la seule façon de savoir si tout ceci vaut quelque chose.

Reproduire : `python -m jjbml.experiences.phase2`
