"""Protocole d'evaluation.

Le protocole qui fait foi est **leave-one-view-out** : trois plis, chacun teste
sur un angle de camera absent de l'entrainement.

Ce n'est pas celui qu'on voulait. On voulait tenir un combat entier a l'ecart,
mais c'est impossible ici : chaque roll de ViCoS est consacre a un petit groupe
de positions (le dos n'existe que dans le roll 5, la tortue que dans le roll 3),
donc ecarter un roll retire ses classes de l'entrainement et le modele doit
deviner des classes jamais vues. `plis_par_roll` reste disponible pour le
montrer, et seulement pour ca.

Consequence a garder en tete pour tout chiffre de cette phase : il mesure la
generalisation a un **nouvel angle de camera**, avec les memes athletes, la
meme salle et les memes instants presents a l'entrainement. C'est un majorant
optimiste. La vraie generalisation ne se mesurera qu'en phase 4, sur des matchs
de competition annotes.

La metrique principale est l'**accuracy equilibree** (rappel moyen par classe),
pas l'accuracy brute : les classes vont de 2 586 a 16 722 exemples, donc
l'accuracy brute recompense qui predit `standing`.

Sous-echantillonnage : a 30 images par seconde, une image sur cinq porte
presque toute l'information d'une seconde. Entrainer sur un pas de 5 divise le
temps par cinq sans rien perdre de substantiel, et ca n'affecte pas
l'evaluation, qui se fait toujours sur la totalite du pli de test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from .data.vicos import CLASSES_18, POSITIONS_10, Jeu
from .parsers.vicos import CLASSES


# indice de classe 18 -> indice de position 10, pour replier les predictions
_VERS_10 = np.array(
    [POSITIONS_10.index(CLASSES[c][0]) for c in CLASSES_18], dtype=np.int64
)


@dataclass
class Resultat:
    nom: str
    acc_equilibree: float
    acc_brute: float
    acc_equilibree_10: float
    par_pli: list[float] = field(default_factory=list)
    matrice: np.ndarray | None = None
    y_vrai: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    classes_jamais_vues: list[int] = field(default_factory=list)

    def __str__(self) -> str:
        plis = " ".join(f"{p:.3f}" for p in self.par_pli)
        texte = (
            f"{self.nom:<30s} equilibree {self.acc_equilibree:.3f}  "
            f"brute {self.acc_brute:.3f}  10 pos {self.acc_equilibree_10:.3f}\n"
            f"{'':<30s} par pli : {plis}"
        )
        if self.classes_jamais_vues:
            noms = ", ".join(CLASSES_18[i] for i in self.classes_jamais_vues)
            texte += (
                f"\n{'':<30s} PROTOCOLE INVALIDE : {len(self.classes_jamais_vues)} "
                f"classes testees sans jamais avoir ete vues ({noms})"
            )
        return texte


def evaluer(
    fabrique,
    jeu: Jeu,
    *,
    nom: str,
    pas: int = 5,
    augmente: bool = False,
    protocole: str = "vue",
) -> Resultat:
    """Evalue sur les 18 classes.

    `fabrique` rend un objet neuf a chaque pli, avec fit(X, y) et predict(X).
    `augmente` ajoute au jeu d'entrainement la version aux athletes echanges.
    `protocole` vaut "vue" (le bon) ou "roll" (celui qui ne peut pas marcher).
    """
    X = jeu.normalise()
    y = jeu.y18
    plis = jeu.plis_par_vue() if protocole == "vue" else jeu.plis_par_roll()

    vrais, predits, par_pli, jamais_vues = [], [], [], set()

    for _, train, test in plis:
        jamais_vues.update(jeu.classes_absentes_du_train(train, test))
        idx = np.flatnonzero(train)[::pas]
        Xtr, ytr = X[idx], y[idx]

        if augmente:
            Xsw, ysw = jeu.echange_athletes(X)
            Xtr = np.concatenate([Xtr, Xsw[idx]])
            ytr = np.concatenate([ytr, ysw[idx]])

        modele = fabrique()
        modele.fit(Xtr, ytr)
        p = modele.predict(X[test])

        vrais.append(y[test])
        predits.append(p)
        par_pli.append(balanced_accuracy_score(y[test], p))

    yv = np.concatenate(vrais)
    yp = np.concatenate(predits)
    return Resultat(
        nom=nom,
        classes_jamais_vues=sorted(jamais_vues),
        acc_equilibree=float(balanced_accuracy_score(yv, yp)),
        acc_brute=float((yv == yp).mean()),
        acc_equilibree_10=float(
            balanced_accuracy_score(_VERS_10[yv], _VERS_10[yp])
        ),
        par_pli=par_pli,
        matrice=confusion_matrix(yv, yp, labels=range(len(CLASSES_18))),
        y_vrai=yv,
        y_pred=yp,
    )


def evaluer_split_naif(fabrique, jeu: Jeu, *, nom: str, pas: int = 5) -> Resultat:
    """Le MAUVAIS protocole, pour mesurer la fuite. Jamais pour conclure."""
    X = jeu.normalise()
    y = jeu.y18
    train, test = jeu.split_naif_par_frame()
    idx = np.flatnonzero(train)[::pas]

    modele = fabrique()
    modele.fit(X[idx], y[idx])
    p = modele.predict(X[test])
    return Resultat(
        nom=nom,
        acc_equilibree=float(balanced_accuracy_score(y[test], p)),
        acc_brute=float((y[test] == p).mean()),
        acc_equilibree_10=float(
            balanced_accuracy_score(_VERS_10[y[test]], _VERS_10[p])
        ),
    )


def afficher_matrice(resultat: Resultat, positions_seulement: bool = True) -> str:
    """Matrice de confusion lisible dans un terminal."""
    if resultat.y_vrai is None:
        return "(pas de matrice)"
    if positions_seulement:
        yv, yp = _VERS_10[resultat.y_vrai], _VERS_10[resultat.y_pred]
        noms = POSITIONS_10
    else:
        yv, yp, noms = resultat.y_vrai, resultat.y_pred, CLASSES_18

    m = confusion_matrix(yv, yp, labels=range(len(noms)))
    m = m / np.clip(m.sum(axis=1, keepdims=True), 1, None) * 100

    court = [n[:9] for n in noms]
    lignes = ["      " + "".join(f"{n[:5]:>6s}" for n in court)]
    for i, n in enumerate(court):
        cellules = "".join(
            f"{m[i, j]:5.0f}{'*' if i == j else ' '}" for j in range(len(noms))
        )
        lignes.append(f"{n:>5s} {cellules}")
    lignes.append("(lignes = verite, colonnes = prediction, en % de la ligne)")
    return "\n".join(lignes)
