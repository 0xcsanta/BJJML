"""Jeu de donnees ViCoS : du JSON aux tableaux, et le split qui ne triche pas.

CE QUE CONTIENT UN EXEMPLE
--------------------------

17 articulations COCO par athlete, chacune (x, y, confiance), pour deux
athletes : 102 caracteristiques. Aucune image n'est necessaire, les keypoints
sont deja dans annotations.json.

...SAUF QUE LES DEUX POSES NE SONT PAS TOUJOURS LA
--------------------------------------------------

Releve sur les donnees, et ce n'est pas documente par la page du dataset :

    76 343 (63,5 %)  pose1 et pose2
    23 398 (19,5 %)  pose1 seule
    20 538 (17,1 %)  pose2 seule

Un tiers des exemples n'ont donc qu'un seul athlete detecte. Et ce n'est pas
aleatoire : `turtle1` perd pose2 3 990 fois, `half_guard2` 2 989 fois. Ce sont
exactement les positions les plus emmelees, ou le detecteur de pose fusionne ou
perd un corps. C'est le probleme d'association de membres, mesure.

On ne jette donc PAS ces exemples. Les jeter retirerait du jeu precisement les
situations difficiles, ce qui gonflerait le score sans rien ameliorer : une
donnee manquante qui depend de la classe n'est jamais ignorable. Une pose
absente est mise a zero et signalee par un indicateur de presence. D'ou
**104 caracteristiques** : 102 + 2 drapeaux.

Note : les confiances ne sont pas bornees a 1 (observe jusqu'a 1,06).

LE SPLIT, QUI EST TOUT L'ENJEU
------------------------------

A 30 images par seconde deux images voisines sont la meme image, et les
16 identifiants de vue ne sont que 6 combats filmes par jusqu'a 3 cameras
(voir jjbml.parsers.vicos). Le seul decoupage honnete est donc **par roll**,
et il n'y en a que six : leave-one-roll-out, six plis, chacun teste sur un
combat jamais vu.

NORMALISATION
-------------

Sans elle, le modele apprend ou se trouve la paire dans le cadre et a quelle
distance filme la camera : des proprietes de cette salle, pas du jiu-jitsu. On
recentre sur le barycentre des articulations presentes et on met a l'echelle
par leur dispersion, pondere par les confiances pour que les detections
douteuses ne tirent pas le repere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..parsers.vicos import (
    CLASSES,
    NB_RANGS,
    POSITIONS,
    RANG_DE_VUE,
    ROLLS,
    VUE_VERS_ROLL,
)

N_JOINTS = 17
N_ATHLETES = 2
N_PAR_ATHLETE = N_JOINTS * 3
N_KEYPOINTS = N_PAR_ATHLETE * N_ATHLETES  # 102
N_FEATURES = N_KEYPOINTS + N_ATHLETES  # 104, avec les drapeaux de presence

CLASSES_18 = tuple(CLASSES)
POSITIONS_10 = tuple(POSITIONS)
_INDEX_18 = {c: i for i, c in enumerate(CLASSES_18)}
_INDEX_10 = {p: i for i, p in enumerate(POSITIONS_10)}

# X1 devient X2 quand on echange les deux athletes, et inversement.
_ECHANGE_18 = np.array(
    [
        _INDEX_18[
            c if CLASSES[c][1] is None else f"{CLASSES[c][0]}{3 - CLASSES[c][1]}"
        ]
        for c in CLASSES_18
    ],
    dtype=np.int64,
)


def chemin_npz() -> Path:
    from ..fetch._common import project_root

    d = project_root() / "data" / "derived"
    d.mkdir(parents=True, exist_ok=True)
    return d / "vicos_keypoints.npz"


def preparer(force: bool = False) -> Path:
    """Convertit annotations.json en tableaux numpy, une fois pour toutes."""
    sortie = chemin_npz()
    if sortie.exists() and not force:
        print(f"  deja prepare : {sortie.name}")
        return sortie

    from ..parsers.vicos import chemin_par_defaut

    source = chemin_par_defaut()
    if not source.exists():
        raise FileNotFoundError(
            f"{source} absent. Lancer d'abord : python -m jjbml.fetch.vicos"
        )

    print(f"  lecture de {source.name} (199 Mo, quelques Go de memoire)...")
    enregistrements = json.loads(source.read_text(encoding="utf-8"))
    n = len(enregistrements)

    X = np.zeros((n, N_KEYPOINTS), dtype=np.float32)
    present = np.zeros((n, N_ATHLETES), dtype=bool)
    y18 = np.zeros(n, dtype=np.int64)
    y10 = np.zeros(n, dtype=np.int64)
    roll = np.zeros(n, dtype=np.int64)
    vue = np.zeros(n, dtype=np.int64)
    frame = np.zeros(n, dtype=np.int64)

    for i, rec in enumerate(enregistrements):
        for a, cle in enumerate(("pose1", "pose2")):
            pose = rec.get(cle)
            if pose is None:
                continue  # laisse des zeros ; le drapeau dira que c'est absent
            debut = a * N_PAR_ATHLETE
            X[i, debut : debut + N_PAR_ATHLETE] = np.asarray(
                pose, dtype=np.float32
            ).reshape(-1)
            present[i, a] = True

        classe = rec["position"]
        y18[i] = _INDEX_18[classe]
        y10[i] = _INDEX_10[CLASSES[classe][0]]
        image = rec["image"]
        vue[i] = int(image[:2])
        roll[i] = VUE_VERS_ROLL[image[:2]]
        frame[i] = int(image[2:])

    complets = int(present.all(axis=1).sum())
    print(f"  {n} exemples, dont {complets} ({100*complets/n:.1f} %) avec les deux poses")

    np.savez_compressed(
        sortie, X=X, present=present, y18=y18, y10=y10, roll=roll, vue=vue, frame=frame
    )
    print(f"  ecrit : {sortie} ({sortie.stat().st_size / 1e6:.0f} Mo)")
    return sortie


@dataclass
class Jeu:
    X: np.ndarray  # (n, 102) brut, zeros pour une pose absente
    present: np.ndarray  # (n, 2) bool
    y18: np.ndarray
    y10: np.ndarray
    roll: np.ndarray
    vue: np.ndarray
    frame: np.ndarray

    def __len__(self) -> int:
        return len(self.y18)

    @property
    def complets(self) -> np.ndarray:
        """Masque des exemples ou les deux athletes ont ete detectes."""
        return self.present.all(axis=1)

    # -- mise en forme ------------------------------------------------------

    def poses(self) -> np.ndarray:
        """(n, 2, 17, 3) : athlete, articulation, (x, y, confiance)."""
        return self.X.reshape(-1, N_ATHLETES, N_JOINTS, 3)

    def normalise(self) -> np.ndarray:
        """(n, 104) : keypoints recentres et mis a l'echelle, puis les drapeaux.

        Le barycentre et l'echelle ne sont calcules que sur les articulations
        effectivement presentes : une pose absente ne doit pas tirer le repere
        vers l'origine.
        """
        p = self.poses().astype(np.float64)
        xy = p[..., :2].reshape(len(p), -1, 2)  # (n, 34, 2)
        conf = p[..., 2].reshape(len(p), -1, 1)  # (n, 34, 1)

        presence = np.repeat(self.present, N_JOINTS, axis=1)[..., None]
        poids = np.where(presence, np.clip(conf, 1e-3, None), 0.0)
        total = np.clip(poids.sum(axis=1, keepdims=True), 1e-6, None)

        centre = (xy * poids).sum(axis=1, keepdims=True) / total
        recentre = np.where(presence, xy - centre, 0.0)

        var = (poids * recentre**2).sum(axis=(1, 2), keepdims=True) / np.clip(
            poids.sum(axis=(1, 2), keepdims=True), 1e-6, None
        )
        recentre = recentre / np.clip(np.sqrt(var), 1e-6, None)

        conf_nette = np.where(presence, conf, 0.0)
        keypoints = np.concatenate([recentre, conf_nette], axis=2).reshape(len(p), -1)
        return np.concatenate(
            [keypoints, self.present.astype(np.float64)], axis=1
        ).astype(np.float32)

    def echange_athletes(self, Xn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Intervertit les deux athletes, etiquettes et drapeaux compris.

        Augmentation : le modele doit lire la geometrie, pas l'ordre dans
        lequel le detecteur a range les deux personnes.
        """
        k, d = N_PAR_ATHLETE, N_KEYPOINTS
        permute = np.concatenate(
            [Xn[:, k:d], Xn[:, :k], Xn[:, d + 1 : d + 2], Xn[:, d : d + 1]], axis=1
        )
        return permute, _ECHANGE_18[self.y18]

    # -- splits -------------------------------------------------------------

    def couverture_par_roll(self) -> dict[int, list[int]]:
        """Quelles classes apparaissent dans quel roll."""
        return {
            r: sorted(np.unique(self.y18[self.roll == r]).tolist())
            for r in range(len(ROLLS))
        }

    def classes_absentes_du_train(self, train, test) -> list[int]:
        """Classes presentes en test mais jamais vues a l'entrainement."""
        vues = set(np.unique(self.y18[train]).tolist())
        return sorted(set(np.unique(self.y18[test]).tolist()) - vues)

    def plis_par_roll(self):
        """Leave-one-roll-out. INUTILISABLE sur ce jeu de donnees.

        Chaque roll est consacre a un petit groupe de positions : le dos
        n'existe que dans le roll 5, la tortue que dans le roll 3, le mount que
        dans le roll 2. Mettre un roll en test retire donc ses classes de
        l'entrainement, et le modele doit predire des classes qu'il n'a jamais
        vues. Les scores s'effondrent a zero sans que cela dise quoi que ce soit
        du modele.

        Conserve pour pouvoir le montrer, jamais pour conclure. Utiliser
        `plis_par_vue`.
        """
        for r in range(len(ROLLS)):
            test = self.roll == r
            yield r, ~test, test

    def plis_par_vue(self):
        """Le seul protocole que ce jeu de donnees peut honnetement supporter.

        On tient a l'ecart une camera par roll : trois plis, chacun teste sur
        un angle de prise de vue absent de l'entrainement. Toutes les classes
        restent presentes des deux cotes.

        CE QUE CELA MESURE, ET SURTOUT CE QUE CELA NE MESURE PAS. C'est une
        generalisation a un **nouvel angle de camera**, pas a un nouveau
        combat : les memes athletes, la meme salle et les memes instants sont
        dans l'entrainement, vus d'ailleurs. Le chiffre obtenu est donc un
        majorant optimiste de ce que le modele vaudrait sur un vrai match.
        Seule la phase 4 permettra de mesurer la vraie generalisation.
        """
        rang = np.array(
            [RANG_DE_VUE[f"{v:02d}"] for v in self.vue], dtype=np.int64
        )
        for k in range(NB_RANGS):
            test = rang == k
            if test.any():
                yield k, ~test, test

    def split_naif_par_frame(self, graine: int = 0, part_test: float = 0.2):
        """Decoupage aleatoire par image. A NE PAS UTILISER pour conclure.

        Present uniquement pour mesurer l'ampleur de la fuite : la frame 1047
        d'entrainement et la 1048 de test sont la meme image, et la meme scene
        existe en plus sous deux autres angles.
        """
        rng = np.random.default_rng(graine)
        test = rng.random(len(self)) < part_test
        return ~test, test


def charger() -> Jeu:
    chemin = chemin_npz()
    if not chemin.exists():
        preparer()
    d = np.load(chemin)
    return Jeu(
        X=d["X"], present=d["present"], y18=d["y18"], y10=d["y10"],
        roll=d["roll"], vue=d["vue"], frame=d["frame"],
    )


if __name__ == "__main__":
    preparer()
