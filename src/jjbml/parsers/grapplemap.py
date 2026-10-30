"""Parser du fichier GrappleMap.txt.

Le format n'est pas documente et son auteur previent qu'il est ad-hoc. Ce qui
suit a ete retro-concu depuis le fichier lui-meme, puis confirme sur le code
source de reference (src/persistence.cpp et src/graph.cpp du depot GrappleMap).

STRUCTURE D'UNE ENTREE
----------------------

    <nom>                        colonne 0, les retours a la ligne sont des \\n litteraux
    tags: a b c                  optionnel
    properties: top              optionnel
    ref: <citation>              optionnel, repetable (source instructionale)
        <4 lignes de 69 caracteres>    un keyframe
        <4 lignes de 69 caracteres>    keyframe suivant, s'il y en a

Une entree a un seul keyframe est une POSITION. Une entree qui en a plusieurs
est une TRANSITION, dont le premier et le dernier keyframe sont les extremites.

ENCODAGE DES POSES
------------------

Confirme sur persistence.cpp : base62 avec l'alphabet "a-zA-Z0-9" (minuscules
d'abord), deux caracteres par valeur, `valeur = (c0 * 62 + c1) / 1000`, puis
`{x: v - 2, y: v, z: v - 2}` articulation par articulation. 23 articulations
par athlete, deux athletes, soit 46 * 3 = 138 valeurs = 276 caracteres,
repartis sur 4 lignes de 69.

RACCORDEMENT DU GRAPHE
----------------------

graph.cpp raccorde chaque transition a ses positions ainsi :

    from = find_or_add(seq->positions.front());
    to   = find_or_add(seq->positions.back());

et `find_or_add` compare les poses **a une reorientation pres** (rotation et
translation). Une comparaison octet pour octet echoue donc : verifie sur ces
donnees, elle ne raccorde que 235 transitions sur 1485. On compare ici des
signatures invariantes par deplacement : la matrice des distances entre
articulations ne change ni par rotation, ni par translation, ni par symetrie.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

B62 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
B62_INDEX = {c: i for i, c in enumerate(B62)}

JOINTS_PAR_ATHLETE = 23
JOINTS_TOTAL = JOINTS_PAR_ATHLETE * 2
VALEURS_PAR_KEYFRAME = JOINTS_TOTAL * 3
CARACTERES_PAR_KEYFRAME = VALEURS_PAR_KEYFRAME * 2
LIGNES_PAR_KEYFRAME = 4

MOTS_CLES = ("tags:", "properties:", "ref:", "todo:", "note:")

# Tolerance d'appariement, en metres, sur l'ecart quadratique moyen entre deux
# matrices de distances. L'encodage quantifie au millimetre, donc l'ecart entre
# deux enregistrements d'une meme pose est de l'ordre de quelques millimetres ;
# 2 cm laisse de la marge sans confondre deux positions distinctes.
TOLERANCE_APPARIEMENT = 0.02


def decode_keyframe(chars: str) -> np.ndarray:
    """Decode 276 caracteres base62 en un tableau (46, 3) de coordonnees.

    Reproduit decodePosition() de persistence.cpp.
    """
    if len(chars) != CARACTERES_PAR_KEYFRAME:
        raise ValueError(
            f"keyframe de {len(chars)} caracteres, {CARACTERES_PAR_KEYFRAME} attendus"
        )
    valeurs = [
        (B62_INDEX[chars[i]] * 62 + B62_INDEX[chars[i + 1]]) / 1000
        for i in range(0, len(chars), 2)
    ]
    pose = np.asarray(valeurs, dtype=np.float64).reshape(JOINTS_TOTAL, 3)
    pose[:, 0] -= 2.0  # x
    pose[:, 2] -= 2.0  # z ; y (hauteur) reste tel quel
    return pose


def signature(pose: np.ndarray) -> np.ndarray:
    """Signature invariante par rotation, translation et symetrie.

    C'est la matrice des distances entre articulations, dont on ne garde que le
    triangle superieur. Deux enregistrements d'une meme pose, orientes
    differemment, donnent la meme signature.
    """
    deltas = pose[:, None, :] - pose[None, :, :]
    distances = np.sqrt((deltas**2).sum(axis=-1))
    haut = np.triu_indices(JOINTS_TOTAL, k=1)
    return distances[haut]


@dataclass
class Entree:
    """Une entree du fichier : position si un keyframe, transition si plusieurs."""

    nom: str
    tags: list[str] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    keyframes: np.ndarray | None = None  # (n, 46, 3)

    @property
    def est_position(self) -> bool:
        return self.keyframes is not None and len(self.keyframes) == 1

    @property
    def est_transition(self) -> bool:
        return self.keyframes is not None and len(self.keyframes) > 1

    @property
    def nom_lisible(self) -> str:
        """Le nom avec ses retours a la ligne litteraux remplaces par des espaces."""
        return re.sub(r"\s+", " ", self.nom.replace("\\n", " ")).strip()


@dataclass
class Arete:
    """Une transition raccordee a ses deux noeuds."""

    transition: int  # index dans Graphe.transitions
    depart: int  # index dans Graphe.noeuds
    arrivee: int


@dataclass
class Graphe:
    """Le graphe complet : tout noeud existe, nomme ou non.

    Mesure faite sur ces donnees : 81 % des extremites de transition tombent a
    moins de 0,6 mm d'une position nommee, et les autres a plus de 16 cm. La
    distribution est franchement bimodale, donc le seuil n'est pas un reglage
    delicat : les 19 % restants ne sont pas des appariements rates, ce sont de
    vraies poses intermediaires sans nom. find_or_add() les cree comme noeuds a
    part entiere, et on fait pareil, sinon le graphe serait troue.
    """

    noeuds: list[Entree]  # d'abord les positions nommees, puis les anonymes
    transitions: list[Entree]
    aretes: list[Arete]
    sans_keyframe: list[Entree]
    n_nommes: int

    @property
    def positions(self) -> list[Entree]:
        """Les noeuds qui portent un nom dans le fichier."""
        return self.noeuds[: self.n_nommes]

    @property
    def anonymes(self) -> list[Entree]:
        """Les noeuds crees pour des poses intermediaires sans nom."""
        return self.noeuds[self.n_nommes :]

    def tags(self) -> dict[str, int]:
        """Les tags et leur frequence, toutes entrees confondues."""
        compte: dict[str, int] = {}
        for e in self.noeuds + self.transitions:
            for t in e.tags:
                compte[t] = compte.get(t, 0) + 1
        return dict(sorted(compte.items(), key=lambda kv: -kv[1]))

    def voisins(self, noeud: int) -> list[tuple[int, Entree]]:
        """Les noeuds atteignables depuis celui-ci, avec la transition qui y mene."""
        return [
            (a.arrivee, self.transitions[a.transition])
            for a in self.aretes
            if a.depart == noeud
        ]


def _lire_entrees(texte: str) -> list[Entree]:
    entrees: list[Entree] = []
    courante: Entree | None = None
    lignes_encodees: list[str] = []

    def cloturer() -> None:
        if courante is None:
            return
        if lignes_encodees:
            n = len(lignes_encodees) // LIGNES_PAR_KEYFRAME
            blocs = [
                "".join(lignes_encodees[i * LIGNES_PAR_KEYFRAME : (i + 1) * LIGNES_PAR_KEYFRAME])
                for i in range(n)
            ]
            courante.keyframes = np.stack([decode_keyframe(b) for b in blocs])

    for brute in texte.splitlines():
        ligne = brute.rstrip()
        if not ligne.strip():
            continue
        if ligne[0].isspace():
            if courante is not None:
                lignes_encodees.append(ligne.strip())
        elif ligne.startswith(MOTS_CLES):
            if courante is not None:
                cle, _, valeur = ligne.partition(":")
                valeur = valeur.strip()
                if cle == "tags":
                    courante.tags.extend(valeur.split())
                elif cle == "properties":
                    courante.properties.extend(valeur.split())
                elif cle == "ref":
                    courante.refs.append(valeur)
        else:
            cloturer()
            courante = Entree(nom=ligne)
            lignes_encodees = []
            entrees.append(courante)

    cloturer()
    return entrees


def _construire_noeuds(
    positions: list[Entree], transitions: list[Entree]
) -> tuple[list[Entree], list[Arete]]:
    """Reproduit find_or_add() de graph.cpp.

    Chaque extremite de transition est cherchee parmi les noeuds connus par
    signature invariante ; si rien ne correspond, elle devient un nouveau noeud
    anonyme. Le graphe rendu n'a donc aucune arete pendante.
    """
    noeuds = list(positions)
    echange = np.roll(np.arange(JOINTS_TOTAL), JOINTS_PAR_ATHLETE)
    anonymes = 0

    # Table des signatures preallouee : la faire grandir par np.stack a chaque
    # ajout rendrait la construction quadratique, soit des minutes au lieu de
    # quelques secondes.
    dim = JOINTS_TOTAL * (JOINTS_TOTAL - 1) // 2
    table = np.empty((len(positions) + 2 * len(transitions), dim))
    for i, p in enumerate(positions):
        table[i] = signature(p.keyframes[0])
    n = len(positions)

    def trouver_ou_ajouter(pose: np.ndarray) -> int:
        nonlocal n, anonymes
        connues = table[:n]
        meilleur, ecart_min = 0, np.inf
        for candidate in (pose, pose[echange]):
            ecarts = np.sqrt(((connues - signature(candidate)) ** 2).mean(axis=1))
            i = int(np.argmin(ecarts))
            if ecarts[i] < ecart_min:
                meilleur, ecart_min = i, float(ecarts[i])
        if ecart_min <= TOLERANCE_APPARIEMENT:
            return meilleur
        anonymes += 1
        noeuds.append(Entree(nom=f"(sans nom {anonymes})", keyframes=pose[None, :, :]))
        table[n] = signature(pose)
        n += 1
        return n - 1

    aretes = [
        Arete(i, trouver_ou_ajouter(t.keyframes[0]), trouver_ou_ajouter(t.keyframes[-1]))
        for i, t in enumerate(transitions)
    ]
    return noeuds, aretes


def chemin_par_defaut() -> Path:
    from ..fetch._common import raw_dir

    return raw_dir("grapplemap") / "GrappleMap.txt"


def load(path: Path | None = None) -> Graphe:
    """Lit GrappleMap.txt et reconstruit le graphe des positions et transitions."""
    path = path or chemin_par_defaut()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} absent. Lancer d'abord : python -m jjbml.fetch.grapplemap"
        )

    entrees = _lire_entrees(path.read_text(encoding="utf-8", errors="replace"))
    positions = [e for e in entrees if e.est_position]
    transitions = [e for e in entrees if e.est_transition]
    sans_kf = [e for e in entrees if e.keyframes is None]

    noeuds, aretes = _construire_noeuds(positions, transitions)
    return Graphe(
        noeuds=noeuds,
        transitions=transitions,
        aretes=aretes,
        sans_keyframe=sans_kf,
        n_nommes=len(positions),
    )
