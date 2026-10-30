"""Parser de graph.json.gz (bjjgraph).

STRUCTURE
---------

    positions     397  chaque hub decline en Top, Bottom et hub neutre
    transitions  3006
    submissions   923
    principles     63
    systems        83
    meta

Une position porte : name, hub, role (top/bottom), path, pointValue,
positionType, riskLevel, energyCost, transitions, flashcards, strength.

RAPPEL, ET CE N'EST PAS UN DETAIL
---------------------------------

`strength`, `pointValue` et les probabilites portees par les transitions sont
des *authored estimates* : ecrites a la main, regenerables par enrichissement
IA, et dont le README de la source dit lui-meme qu'elles attendent une revue
par un panel de ceintures noires qui n'a pas eu lieu.

On ne prend d'ici que la STRUCTURE. Les valeurs servent de point de comparaison
pour nos mesures (PLAN.md, phase 6), jamais de verite terrain. La fonction
`valeurs_estimees()` porte ce nom pour qu'on ne puisse pas s'y tromper.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path


def chemin_par_defaut() -> Path:
    from ..fetch._common import raw_dir

    return raw_dir("bjjgraph") / "graph.json.gz"


def load(path: Path | None = None) -> dict:
    path = path or chemin_par_defaut()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} absent. Lancer d'abord : python -m jjbml.fetch.bjjgraph"
        )
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def hubs(donnees: dict | None = None) -> dict[str, dict[str, dict]]:
    """Regroupe les 397 positions par hub : {hub: {role: position}}.

    Le role vaut 'top', 'bottom' ou 'hub' pour l'entree neutre.
    """
    donnees = donnees if donnees is not None else load()
    groupes: dict[str, dict[str, dict]] = {}
    for cle, p in donnees["positions"].items():
        groupes.setdefault(p.get("hub") or cle, {})[p.get("role") or "hub"] = p
    return groupes


def valeurs_estimees(donnees: dict | None = None) -> dict[str, float]:
    """Le `strength` du cote dessus, par hub.

    ESTIMATIONS ECRITES A LA MAIN, PAS DES MESURES. A n'utiliser que pour
    comparer avec nos propres valeurs mesurees.
    """
    return {
        h: roles["top"]["strength"]
        for h, roles in hubs(donnees).items()
        if "top" in roles and roles["top"].get("strength") is not None
    }
