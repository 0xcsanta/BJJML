"""Reproduit les chiffres de results/phase2.md.

    python -m jjbml.experiences.phase2

Compter une dizaine de minutes sur CPU.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from ..data import vicos as dv
from .. import evaluation as ev
from ..models import baselines as bl
from ..models.mlp import MLP

warnings.filterwarnings("ignore")


def main() -> None:
    jeu = dv.charger()
    print(f"{len(jeu)} exemples, {int(jeu.complets.sum())} avec les deux poses\n")

    print("=" * 70)
    print("1. Pourquoi le leave-one-roll-out ne peut pas marcher ici")
    print("=" * 70)
    couverture = jeu.couverture_par_roll()
    for r, classes in couverture.items():
        noms = ", ".join(dv.CLASSES_18[c] for c in classes)
        print(f"  roll {r} : {len(classes)} classes -> {noms}")
    print()
    print(ev.evaluer(bl.regression_logistique, jeu,
                     nom="reg. logistique (par roll)", protocole="roll"))

    print()
    print("=" * 70)
    print("2. Leave-one-view-out : nouvel angle de camera, meme combat")
    print("=" * 70)
    resultats = [
        ev.evaluer(bl.ClasseMajoritaire, jeu, nom="classe majoritaire"),
        ev.evaluer(bl.regression_logistique, jeu, nom="regression logistique"),
        ev.evaluer(lambda: MLP(), jeu, nom="MLP 256-128"),
        ev.evaluer(lambda: MLP(), jeu, nom="MLP 256-128 + echange", augmente=True),
    ]
    for r in resultats:
        print(r)
        print()

    meilleur = resultats[-1]

    print("=" * 70)
    print("3. Ce qu'aurait donne un split malhonnete")
    print("=" * 70)
    fuite = ev.evaluer_split_naif(lambda: MLP(), jeu, nom="MLP, split par image")
    print(fuite)
    print(f"\n  fuite mesuree : +{fuite.acc_equilibree - meilleur.acc_equilibree:.3f}")

    print()
    print("=" * 70)
    print("4. Ou le modele se trompe")
    print("=" * 70)
    print(ev.afficher_matrice(meilleur))

    print()
    print("=" * 70)
    print("5. Le vrai plafond : une pose ou deux")
    print("=" * 70)
    ordre = np.concatenate([np.flatnonzero(t) for _, _, t in jeu.plis_par_vue()])
    complets = jeu.complets[ordre]
    for nom, masque in (("les deux athletes", complets), ("un seul athlete", ~complets)):
        score = balanced_accuracy_score(
            meilleur.y_vrai[masque], meilleur.y_pred[masque]
        )
        print(f"  {nom:20s} n={masque.sum():6d}  equilibree {score:.3f}")
    print("\n  Le facteur limitant est l'estimation de pose, pas le classifieur.")


if __name__ == "__main__":
    main()
