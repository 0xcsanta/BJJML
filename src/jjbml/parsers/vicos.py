"""Parser des annotations ViCoS, et surtout : le bon groupe de split.

FORMAT REEL
-----------

annotations.json est un tableau d'objets aux cles **en minuscules**, alors que
la page du dataset les documente capitalisees (Image, Pose1, Position, Frame) :

    {"position": "standing", "image": "0000001", "frame": 1, "pose1": [...], "pose2": [...]}

`image` vaut 2 chiffres d'identifiant de vue suivis de 5 chiffres de frame.
Chaque pose est une liste de 17 articulations [x, y, confiance], au format COCO.

Les 18 classes sont 8 positions declinees en deux roles (le suffixe 1 ou 2
designe l'athlete), plus `standing` et `5050_guard` qui n'en ont qu'une.

LE PIEGE DU SPLIT
-----------------

Le champ `image` porte **16 identifiants**, mais ce ne sont pas 16 rolls : la
page annonce 6 sequences de sparring filmees par 3 cameras. Verifie sur les
donnees en comparant les timelines de labels frame par frame, l'accord est
binaire : 98 a 100 % entre deux vues d'un meme roll, 0 a 5 % entre deux rolls
differents. Les composantes connexes donnent exactement **6 rolls**, ce qui
confirme la page.

Consequence : splitter sur les deux premiers chiffres de `image` ferait passer
*le meme instant du meme combat, filme d'un autre angle*, du jeu
d'entrainement au jeu de test. L'accuracy serait excellente et ne voudrait rien
dire. **Le groupe de split est le roll, pas la vue.**

Et 6 groupes, c'est peu. C'est la vraie mesure de la diversite de ce jeu de
donnees : pas 120 279 exemples, six combats.
"""

from __future__ import annotations

import json
from pathlib import Path

# Rolls reconstruits par composantes connexes sur l'accord des timelines
# (seuil 80 % ; deux vues d'un meme roll sont a 98-100 %, deux rolls
# differents a 0-5 %, donc le seuil est sans ambiguite).
ROLLS: tuple[tuple[str, ...], ...] = (
    ("00", "01", "02"),
    ("03", "04", "05"),
    ("06", "07", "08"),
    ("09", "10"),
    ("11", "12", "13"),
    ("14", "15"),
)

VUE_VERS_ROLL: dict[str, int] = {
    vue: i for i, vues in enumerate(ROLLS) for vue in vues
}

# Rang de la camera a l'interieur de son roll : 0, 1 ou 2.
# Sert au seul protocole d'evaluation que ce jeu de donnees peut supporter
# (voir jjbml.data.vicos), celui qui tient une camera a l'ecart.
RANG_DE_VUE: dict[str, int] = {
    vue: rang for vues in ROLLS for rang, vue in enumerate(vues)
}
NB_RANGS = max(len(v) for v in ROLLS)

# Les 18 classes, et la position de fond que chacune designe.
CLASSES: dict[str, tuple[str, int | None]] = {
    "standing": ("standing", None),
    "5050_guard": ("5050_guard", None),
    "takedown1": ("takedown", 1),
    "takedown2": ("takedown", 2),
    "open_guard1": ("open_guard", 1),
    "open_guard2": ("open_guard", 2),
    "closed_guard1": ("closed_guard", 1),
    "closed_guard2": ("closed_guard", 2),
    "half_guard1": ("half_guard", 1),
    "half_guard2": ("half_guard", 2),
    "side_control1": ("side_control", 1),
    "side_control2": ("side_control", 2),
    "mount1": ("mount", 1),
    "mount2": ("mount", 2),
    "back1": ("back", 1),
    "back2": ("back", 2),
    "turtle1": ("turtle", 1),
    "turtle2": ("turtle", 2),
}

POSITIONS = tuple(dict.fromkeys(p for p, _ in CLASSES.values()))


def vue(image: str) -> str:
    """Identifiant de vue (camera) : les deux premiers chiffres de `image`."""
    return image[:2]


def roll(image: str) -> int:
    """Identifiant de roll : c'est CE groupe qui sert au split, pas la vue."""
    return VUE_VERS_ROLL[vue(image)]


def numero_de_frame(image: str) -> int:
    return int(image[2:])


def chemin_par_defaut() -> Path:
    from ..fetch._common import raw_dir

    return raw_dir("vicos") / "annotations.json"


def load(path: Path | None = None) -> list[dict]:
    """Charge les annotations (199 Mo : compter quelques Go de memoire)."""
    path = path or chemin_par_defaut()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} absent. Lancer d'abord : python -m jjbml.fetch.vicos"
        )
    return json.loads(path.read_text(encoding="utf-8"))
