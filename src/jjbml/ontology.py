"""L'ontologie : le pont entre les trois vocabulaires.

C'est la piece qui permet de dire qu'une position nommee dans une source est
la meme que dans les deux autres. Elle se charge depuis ontology/*.yaml et
**refuse de demarrer si la table est incoherente** : un nom range sous deux
positions, une classe ViCoS oubliee, une valeur hors de [0, 1]. Mieux vaut une
erreur au chargement qu'un moteur qui conseille a cote.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

SOURCES = ("vicos", "bjjgraph", "grapplemap")


def dossier_ontologie() -> Path:
    return Path(__file__).resolve().parents[2] / "ontology"


class OntologieInvalide(ValueError):
    """La table est incoherente : on prefere echouer tot et bruyamment."""


@dataclass
class Position:
    cle: str
    nom: str
    roles: list[str]
    description: str
    affinements: list[str]


@dataclass
class Ontologie:
    positions: dict[str, Position]
    correspondances: dict[str, dict[str, list[str]]]
    hierarchie: dict[str, dict]
    modificateurs: dict[str, dict[str, str]]
    non_couvert: list[dict]
    _index: dict[tuple[str, str], str]

    # -- resolution ---------------------------------------------------------

    def resoudre(self, nom: str, source: str | None = None) -> str | None:
        """Nom d'une source -> cle canonique. None si inconnu.

        Sans `source`, cherche dans les trois.
        """
        sources = (source,) if source else SOURCES
        for s in sources:
            cle = self._index.get((s, nom.strip().lower()))
            if cle:
                return cle
        return None

    def vers(self, cle: str, source: str) -> list[str]:
        """Cle canonique -> les noms correspondants dans une source."""
        if source not in SOURCES:
            raise ValueError(f"source inconnue : {source}")
        return list(self.correspondances[cle].get(source, []))

    def traduire(self, nom: str, vers_source: str) -> list[str]:
        """Nom d'une source -> noms equivalents dans une autre."""
        cle = self.resoudre(nom)
        return self.vers(cle, vers_source) if cle else []

    # -- valeurs ------------------------------------------------------------

    def valeur(self, cle: str, role: str) -> float:
        """Valeur positionnelle dans [0, 1], du point de vue du role."""
        entree = self.hierarchie[cle]
        if role not in entree:
            raise KeyError(f"role {role!r} absent pour {cle!r} (roles : {self.positions[cle].roles})")
        return float(entree[role])

    def estimation_bjjgraph(self, cle: str) -> float | None:
        """Pour comparaison seulement : estimation ecrite a la main, non mesuree."""
        return self.hierarchie[cle].get("bjjgraph_strength")

    @property
    def est_brouillon(self) -> bool:
        return self.hierarchie.get("_etat") == "brouillon"


def _charger_yaml(chemin: Path) -> dict:
    if not chemin.exists():
        raise FileNotFoundError(f"{chemin} absent")
    return yaml.safe_load(chemin.read_text(encoding="utf-8"))


def _valider(
    positions: dict[str, Position],
    correspondances: dict[str, dict[str, list[str]]],
    hierarchie: dict[str, dict],
) -> list[str]:
    problemes: list[str] = []

    for cle in positions:
        if cle not in correspondances:
            problemes.append(f"{cle} : absent de mapping.yaml")
        if cle not in hierarchie:
            problemes.append(f"{cle} : absent de hierarchy.yaml")

    for cle in correspondances:
        if cle not in positions:
            problemes.append(f"{cle} : dans mapping.yaml mais pas dans positions.yaml")

    # un nom de source ne peut appartenir qu'a une seule position canonique
    for source in SOURCES:
        vu: dict[str, str] = {}
        for cle, corr in correspondances.items():
            for nom in corr.get(source, []):
                n = str(nom).strip().lower()
                if n in vu:
                    problemes.append(
                        f"{source}:{nom} range a la fois sous {vu[n]} et {cle}"
                    )
                vu[n] = cle

    # toutes les classes ViCoS doivent etre couvertes
    try:
        from .parsers.vicos import CLASSES

        mappees = {
            str(n).lower() for c in correspondances.values() for n in c.get("vicos", [])
        }
        for classe in CLASSES:
            if classe.lower() not in mappees:
                problemes.append(f"classe ViCoS non mappee : {classe}")
    except ImportError:  # pragma: no cover
        pass

    # les roles et les valeurs
    for cle, pos in positions.items():
        entree = hierarchie.get(cle, {})
        for role in pos.roles:
            if role not in entree:
                problemes.append(f"{cle} : valeur manquante pour le role {role}")
            else:
                v = entree[role]
                if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                    problemes.append(f"{cle}.{role} : valeur {v!r} hors de [0, 1]")

    return problemes


def load(dossier: Path | None = None) -> Ontologie:
    dossier = dossier or dossier_ontologie()

    brut_positions = _charger_yaml(dossier / "positions.yaml")
    brut_mapping = _charger_yaml(dossier / "mapping.yaml")
    brut_hierarchie = _charger_yaml(dossier / "hierarchy.yaml")

    positions = {
        cle: Position(
            cle=cle,
            nom=p.get("nom", cle),
            roles=list(p.get("roles", [])),
            description=p.get("description", ""),
            affinements=list(p.get("affinements", [])),
        )
        for cle, p in brut_positions["positions"].items()
    }

    correspondances = {
        cle: {s: [str(n) for n in corr.get(s, []) or []] for s in SOURCES}
        for cle, corr in brut_mapping["correspondances"].items()
    }

    hierarchie = dict(brut_hierarchie["valeurs"])
    hierarchie["_etat"] = brut_hierarchie.get("etat")

    problemes = _valider(positions, correspondances, hierarchie)
    if problemes:
        raise OntologieInvalide(
            "ontologie incoherente :\n  - " + "\n  - ".join(problemes)
        )

    index = {
        (s, nom.strip().lower()): cle
        for cle, corr in correspondances.items()
        for s in SOURCES
        for nom in corr[s]
    }

    return Ontologie(
        positions=positions,
        correspondances=correspondances,
        hierarchie=hierarchie,
        modificateurs=brut_positions.get("modificateurs", {}),
        non_couvert=brut_positions.get("non_couvert", []),
        _index=index,
    )
