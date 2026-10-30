# jjb-ML — Plan du projet

> Un moteur de JJB façon **livre d'ouvertures aux échecs** : on lui donne une situation entre deux
> combattants, il recommande le meilleur coup pour un joueur donné et les enchaînements probables
> selon la réaction de l'adversaire.

Dernière mise à jour : 2026-10-30

---

## 0. Décisions actées

| Sujet | Décision | Pourquoi |
|---|---|---|
| Approche | Moteur **statistique**, pas d'AlphaZero | Aucun simulateur physique de grappling n'existe → pas de self-play possible |
| Entrée | **Vidéo**, pas image fixe | Une image seule ne distingue pas une position qui s'améliore d'une qui s'effondre |
| Rôle de l'utilisateur | Il **déclare** la position et qui il est ; la perception propose, il corrige | Découple la décision de la perception : le moteur délivre même si la vision est imparfaite, et chaque correction devient une donnée annotée |
| Ruleset | **No-gi** | Cohérent sur toute la pile : ViCoS est en no-gi, bjjgraph a des valeurs no-gi séparées |
| Objectif | **Apprendre le ML de bout en bout** | Pas un produit → licences non-commerciales sans problème, on privilégie ce qui enseigne |
| Fonction de valeur | Hiérarchie positionnelle **définie à la main** | Pas de système de points IBJJF en no-gi |

**Hors scope, explicitement :** self-play / RL, simulation physique, gi, MMA, temps réel, application mobile.

---

## 1. Architecture en 3 étages

```
VIDEO / PHOTO                       L'UTILISATEUR DECLARE
  |                                 "il est en side control sur moi"
  v                                           |
[ ETAGE A - PERCEPTION ]                      |
  detection 2 personnes -> keypoints          |
  -> tracking -> classifieur temporel         |
  PROPOSE un etat, l'utilisateur CORRIGE      |
  |                                           |
  +------------------> ETAT <-----------------+
                        |
                        v
[ ETAGE B - LE GRAPHE ]   l'ossature, recuperee et non inventee
  positions = noeuds, transitions = aretes, vocabulaire unifie des 3 sources
  |
  v
[ ETAGE C - LE MOTEUR ]   on pese les aretes, puis on cherche
  stats mesurees (frequence, taux de reussite, position d'arrivee si echec)
  + hierarchie positionnelle -> recherche dans l'arbre -> recommandation + enchainements
```

La clé du projet : **l'étage B fournit les arêtes gratuitement, l'étage A permet de les peser à
l'échelle, l'étage C est le livre d'ouvertures.** Personne n'a jamais mesuré les poids.

### L'entrée déclarée : deux chemins vers l'état, et un volant d'inertie

L'utilisateur ne se contente pas de déposer une vidéo. Il dit aussi **qui il est** dans l'image et
**quelle position il subit ou occupe**. Trois conséquences, toutes bonnes :

1. **L'étage A cesse d'être bloquant.** Le moteur rend un service utile même avec une perception
   médiocre, voire absente. La vision devient une commodité qui fait gagner du temps, pas une
   dépendance dont tout le reste dépend. Le projet ne peut plus échouer entièrement sur la vision.
2. **L'utilisateur fournit l'information invisible.** Les grips, le bras piégé, l'underhook, la
   pression — tout ce qu'une caméra ne voit pas depuis un seul angle, la personne qui vit la position
   le sait. C'est la réponse directe au problème de granularité de ViCoS (10 positions, c'est trop
   grossier pour conseiller).
3. **Chaque correction est un exemple annoté, gratuit.** Le modèle propose « closed guard », la
   personne corrige en « half guard » : voilà une image labellisée, dans une salle nouvelle, avec des
   gens nouveaux, sous un angle nouveau. **L'interface qui rend le produit utilisable est aussi la
   chaîne d'annotation qui rend le modèle bon.** C'est la parade la plus solide au problème de
   généralisation de ViCoS.

**Le piège à ne pas tomber dedans :** si on entraîne sur les confirmations, le modèle apprend ses
propres erreurs (quand la personne clique OK sans regarder). On n'entraîne donc **que sur les
corrections** et sur des audits tirés au hasard, jamais sur les confirmations passives. Et la position
déclarée par un débutant peut être fausse : le désaccord entre le modèle et l'utilisateur est en
soi un signal, à conserver plutôt qu'à trancher.

---

## 2. Sources de données

| Source | Contenu | Licence | Rôle |
|---|---|---|---|
| [ViCoS BJJ Positions](https://vicos.si/resources/jiujitsu/) | 120 279 images, 6 séquences de sparring no-gi, 10 positions / 18 classes, keypoints 2D COCO 17 joints fournis | **CC BY-NC-SA 4.0** | Amorçage de l'étage A |
| [GrappleMap](https://github.com/Eelis/GrappleMap) | 39 098 lignes, 919 entrées taguées, 161 tags fins (grips inclus), coordonnées d'articulations | **Domaine public** | Géométrie + vocabulaire des grips |
| [bjjgraph](https://github.com/diogoseca/bjjgraph) | 133 positions, 1 025 transitions, 353 soumissions, JSON | **PolyForm Noncommercial 1.0.0** | Ontologie navigable (structure uniquement) |
| Vidéos de compétition no-gi (ADCC, WNO, Polaris) | À annoter nous-mêmes | Droit d'auteur tiers | Les vraies statistiques |

### Piège à ne jamais oublier

Les probabilités de bjjgraph sont des **« authored estimates »** — écrites à la main, régénérables
via enrichissement IA, et le README dit lui-même que la revue par un panel de ceintures noires
**n'a pas eu lieu**. Ce ne sont pas des données, ce sont des opinions chiffrées.

→ On prend **la structure** du graphe. Jamais les nombres comme vérité terrain.
→ Leurs probas deviennent une **baseline à battre** : « voilà ce qu'un pratiquant estime à la main,
voilà ce que j'ai mesuré sur 50 matchs, voilà les écarts. » C'est un résultat d'évaluation, pas une entrée.

---

## 3. Conformité — règles dures, vérifiées à chaque commit

Le projet mélange trois licences incompatibles entre elles et des vidéos sous droit d'auteur tiers.
La règle qui résout tout : **aucune donnée tierce n'entre dans Git. Jamais.**

### Règles permanentes

1. **`data/` est intégralement gitignoré.** Pas d'images ViCoS, pas de `graph.json`, pas de
   `GrappleMap.txt`, pas de frames, pas de vidéos. Chaque source a un script de téléchargement dans
   `src/jjbml/fetch/`. Le dépôt contient la recette, jamais les ingrédients.
2. **Pas de vidéo YouTube ni de frames extraites dans le dépôt, sous aucune forme.** Les annotations
   de compétition ne stockent que `video_id + timestamp + label` — des faits sur un match, pas de
   l'œuvre. Un fait n'est pas protégeable ; une frame l'est.
3. **`ATTRIBUTIONS.md` tenu à jour**, une entrée par source : nom, URL, licence, date de récupération,
   et ce qu'on en fait exactement.
4. **Citation ViCoS dans le README dès le premier commit** : Hudovernik & Skočaj, *Video-Based
   Detection of Combat Positions and Automatic Scoring in Jiu-jitsu*, MMSports'22.
5. **Le projet est non-commercial**, point. PolyForm NC et CC BY-NC-SA l'imposent tous les deux.
   C'est assumé dans le README, pas subi. **Licence du dépôt : PolyForm Noncommercial 1.0.0**
   (décidé le 2026-10-30).
6. **Poids de modèles entraînés sur ViCoS** : si publiés, sous CC BY-NC-SA 4.0 (le statut juridique
   des poids est flou — on prend la lecture prudente).
7. **Pas de scraping de bjjgraph.org.** On clone le dépôt, qui est fait pour ça.

### Checklist avant chaque push

```
[ ] git status ne montre aucun fichier sous data/
[ ] aucun fichier > 1 Mo ajoute (garde-fou contre un dataset commit par accident)
[ ] ATTRIBUTIONS.md couvre toute source ajoutee depuis le dernier push
[ ] aucune image, frame ou video dans le diff
[ ] le README porte toujours la citation ViCoS
```

À automatiser en hook pre-commit dès la phase 0.

---

## 4. Les phases

Chaque phase produit quelque chose qui marche tout seul. Les estimations sont indicatives.

### Phase 0 — Fondations et conformité `~1 session`

- Repo Git, structure de dossiers, `.gitignore` strict, `README.md` avec citations, `ATTRIBUTIONS.md`,
  `LICENSE`
- Hook pre-commit qui applique la checklist de la section 3
- Un script de fetch par source, idempotent, dans `src/jjbml/fetch/`
- Environnement Python (venv, requirements)

**Fini quand :** `python -m jjbml.fetch.all` télécharge les 3 sources dans `data/raw/`, et `git status`
reste propre.

### Phase 1 — Parser GrappleMap + ontologie unifiée `~2 sessions`

Le schéma de GrappleMap n'est pas documenté et l'auteur prévient qu'il est ad-hoc. Ce qu'on sait déjà :
919 lignes `tags:`, 1 208 `ref:`, 1 244 `properties:`, 302 `todo:`, coordonnées encodées à raison de
4 lignes par keyframe (position = 1 keyframe, transition = plusieurs).

- Écrire le parser, décoder les coordonnées, reconstruire le graphe
- **`ontology/mapping.yaml`** : la table de correspondance entre les 10 classes ViCoS, les 133 positions
  bjjgraph et les 161 tags GrappleMap. C'est du travail de jugement JJB, pas de code.
- **`ontology/hierarchy.yaml`** : valeur positionnelle de 0 à 1 par position (dos pris avec crochets
  ≈ 0.95, mount ≈ 0.85, … dos pris dessous ≈ 0.05). Remplace le système de points qu'on n'a pas.

**Fini quand :** le graphe se charge en mémoire, et une position nommée dans l'une des trois sources se
résout dans les deux autres.

**Ce que tu apprends :** modélisation de données, et le fait qu'en ML le travail d'ontologie précède
toujours le modèle.

### Phase 1.5 — Le moteur à vide `~1 session`

Grâce à l'entrée déclarée, le moteur est démontrable **avant** que la perception existe. On branche
l'étage C directement sur une position saisie à la main.

- Interface minimale (CLI, puis petit formulaire web) : « je suis dessous », « side control », go
- Le graphe de l'étage B fournit les options ; la hiérarchie positionnelle les classe
- Les probabilités de bjjgraph servent de **remplissage provisoire, affiché comme tel** : « estimation
  non mesurée ». La phase 5 les remplacera par des mesures, et la phase 6 comparera les deux.

**Fini quand :** tu tapes une position et tu obtiens des options classées avec l'arbre des suites.
Le produit existe, il est juste encore mal informé.

**Pourquoi maintenant :** ça prouve la boucle de bout en bout très tôt, ça rend le projet montrable, et
ça garantit que même si la vision ne marche jamais bien, tu as quelque chose qui fonctionne.

### Phase 2 — Premier modèle : classifieur de position `~2 sessions`

Entrée : les keypoints ViCoS, déjà fournis. 17 joints × (x, y, confiance) × 2 athlètes = **102 features**.
Sortie : 1 des 10 positions. Un petit MLP suffit, ça tourne sur CPU.

- **Le split est le cœur de l'exercice : par séquence, jamais par frame.** Deux frames consécutives à
  30 fps sont la même image ; un split aléatoire donnerait 99 % d'accuracy qui ne veut rien dire.
  En pratique, le champ `image` d'`annotations.json` vaut par exemple `"0000001"` : **2 chiffres
  d'identifiant de vue puis 5 de frame.** Attention, le piège est plus subtil qu'il n'y paraît : il y a
  **16 vues mais seulement 6 combats**, parce que chaque roll a été filmé par jusqu'à 3 caméras.
  Vérifié en comparant les timelines de labels frame par frame — l'accord est de 98 à 100 % entre deux
  vues d'un même roll, et de 0 à 5 % entre deux rolls différents. **Le groupe de split est donc le
  roll, pas la vue** (`jjbml.parsers.vicos.roll()`), sinon le même instant du même combat, filmé d'un
  autre angle, passe de l'entraînement au test. Et 6 groupes, c'est la vraie mesure de la diversité
  de ce jeu de données : pas 120 279 exemples, six combats.
- Baselines obligatoires avant le réseau : classe majoritaire, puis régression logistique
- Métriques : accuracy équilibrée + matrice de confusion (les classes sont très déséquilibrées)
- Normalisation des poses (centrer, mettre à l'échelle) — sinon le modèle apprend la position dans le cadre
- Lire le papier MMSports'22 pour fixer la cible à atteindre

**Fini quand :** le réseau bat la régression logistique sur des **séquences jamais vues**, et tu sais
expliquer chaque case de la matrice de confusion.

**Ce que tu apprends :** la fuite de données, le vrai sens d'un split, le déséquilibre de classes.
C'est la leçon la plus importante du projet et elle arrive tôt.

### Phase 3 — Temporel : de la frame à la timeline `~2 sessions`

Une prédiction par frame est bruitée et saute. Un match est une **suite d'états stables**.

- Fenêtre glissante de poses → modèle temporel (de la moyenne mobile au 1D-CNN / GRU)
- Lissage et détection des changements de position
- Sortie : la timeline, format de sortie officiel de l'étage A

**Fini quand :** sur une séquence ViCoS de test, la timeline prédite ressemble à la timeline réelle
(peu de transitions fantômes).

**Ce que tu apprends :** modélisation séquentielle, et qu'une bonne sortie par frame ne fait pas une
bonne sortie par match.

### Phase 4 — Annotation de compétition + généralisation `~10 h d'annotation`

ViCoS, c'est 6 séquences dans une seule salle avec une poignée d'athlètes : le modèle de la phase 2
connaît *ces gens-là, dans cette salle-là*. On sort de là.

- Petit outil d'annotation : vidéo + une touche de clavier par position, sortie `video_id, t, position, rôle`
- Annoter ~50 matchs no-gi de haut niveau (~20-40 événements par match, ~15 min par match)
- Fine-tuner le classifieur sur cette nouvelle distribution

**Ces timelines servent deux fois** : elles fine-tunent l'étage A *et* elles pèsent les arêtes de l'étage C.
Une seule session d'annotation, deux problèmes résolus.

**Fini quand :** le classifieur tient la route sur des matchs de compétition jamais vus, et la base
d'annotations est publiable (timestamps seulement, cf. section 3).

**Ce que tu apprends :** le décalage de distribution, le transfer learning, et la collecte de données —
la partie du ML qu'aucun cours ne montre.

### Phase 5 — Le moteur `~3 sessions`

- Depuis les timelines : matrice de transitions réelle. Pour chaque arête : fréquence de tentative,
  taux de réussite, **et où tu atterris en cas d'échec**
- Ce dernier point rend la recommandation honnête : un sweep qui marche 4 fois sur 10 mais qui te fait
  perdre le dos quand il rate n'est pas un bon coup
- Recherche dans l'arbre (expectimax sur quelques coups) avec la hiérarchie positionnelle comme valeur
- Les enchaînements = les meilleures branches de l'arbre, avec la réaction adverse en nœud de hasard

**Fini quand :** `moteur.conseille(position="side_control", role="dessous")` renvoie des options classées
avec leur taux mesuré et l'arbre des suites.

**Ce que tu apprends :** décision séquentielle, estimation sous incertitude, et pourquoi une
recommandation sans le risque associé ne vaut rien.

### Phase 6 — Évaluation `~1 session`

- Comparer les poids mesurés aux estimations de bjjgraph → où un pratiquant se trompe-t-il, et dans quel sens ?
- Validation humaine : soumettre N recommandations à un coach ou une ceinture avancée
- Écrire les limites honnêtement

**Fini quand :** tu peux dire ce que ton moteur sait, ce qu'il ignore, et sur quoi il se trompe.

### Phase 7 — L'interface complète et le volant d'inertie `~2 sessions`

Reprend l'interface minimale de la phase 1.5 et la referme sur elle-même.

- Dépôt d'une vidéo ou d'une photo, désignation de soi et de l'adversaire
- La perception **propose** un état, l'utilisateur **confirme ou corrige**
- Saisie optionnelle de ce que la caméra ne voit pas (grips, bras piégé, underhook)
- Les corrections sont stockées comme exemples annotés et réinjectées à l'entraînement

**Fini quand :** une correction faite dans l'interface se retrouve dans le jeu d'entraînement du
classifieur, et un réentraînement l'utilise.

**Ce que tu apprends :** la boucle de rétroaction humaine, et comment ne pas s'empoisonner avec —
n'entraîner que sur les corrections et les audits aléatoires, jamais sur les confirmations passives.

**Peut être tiré plus tôt** si tu veux commencer à récolter des corrections pendant les phases 4 et 5.

---

## 5. Structure du dépôt

```
jjb-ML/
  README.md              citations + statut non-commercial
  LICENSE
  ATTRIBUTIONS.md        une entree par source tierce
  PLAN.md                ce fichier
  .gitignore             data/ en entier
  requirements.txt

  data/                  JAMAIS COMMIT
    raw/                 vicos/ grapplemap/ bjjgraph/
    derived/

  ontology/
    positions.yaml       notre taxonomie
    mapping.yaml         ViCoS <-> bjjgraph <-> GrappleMap
    hierarchy.yaml       valeur positionnelle 0-1

  annotations/           NOS donnees, commitables (timestamps seulement)
    competition/

  src/jjbml/
    fetch/               un script par source
    parsers/             grapplemap.py bjjgraph.py vicos.py
    data/                datasets, splits par sequence
    models/
    engine/              graphe + recherche
    annotate/            outil d'annotation

  notebooks/
  tests/
```

---

## 6. Stack

Python 3.13, PyTorch (CPU suffisant jusqu'à la phase 3), NumPy / pandas, scikit-learn pour les
baselines, PyYAML, pytest. Plus tard : RTMPose ou ViTPose pour les keypoints sur vidéo brute.

**Pas de GPU NVIDIA détecté sur la machine.** Jusqu'à la phase 3 c'est sans conséquence (un MLP sur
102 features s'entraîne en secondes). À partir de la phase 4, Google Colab pour le fine-tuning.

---

## 7. Risques

| Risque | Gravité | Parade |
|---|---|---|
| Schéma GrappleMap non documenté, parser coûteux | Moyenne | Borné à la phase 1 ; si ça bloque, bjjgraph seul suffit pour démarrer |
| Le modèle ViCoS ne généralise pas | **Élevée, attendue** | C'est tout l'objet de la phase 4 ; ne jamais annoncer les chiffres ViCoS comme une réussite |
| Lassitude d'annotation | Élevée | Outil à raccourcis clavier, par lots de 10 matchs, et chaque lot améliore déjà les stats |
| Les grips sont invisibles en vidéo | Moyenne | Rester au niveau position en v1 ; les grips sont un objectif de phase 5+, pas une hypothèse |
| Keypoints bruités sur corps emmêlés | Moyenne | Problème documenté (association de membres) ; concevoir les étages suivants pour tolérer le bruit |
| Biais de confirmation via l'interface | **Élevée** | N'entraîner que sur les corrections et des audits tirés au hasard, jamais sur les confirmations passives |
| Position déclarée fausse (débutant) | Moyenne | Garder le désaccord modèle / utilisateur comme signal au lieu de le trancher ; ne pas traiter la déclaration comme une vérité absolue |

---

## 8. Par où on commence

Phase 0, puis phase 1. Le premier vrai jalon de ML est la **phase 2** — et il arrive vite, parce que
les keypoints ViCoS sont déjà fournis : pas besoin de résoudre la vision avant d'entraîner un premier
réseau.
