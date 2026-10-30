"""Telecharge les trois sources du projet.

    python -m jjbml.fetch.all

Tout atterrit sous data/, qui est gitignore. Les 9,35 Go d'images ViCoS ne sont
pas pris : ils sont inutiles jusqu'a la phase 4 (les keypoints sont dans
annotations.json). Pour les avoir : python -m jjbml.fetch.vicos --images
"""

from __future__ import annotations

from . import bjjgraph, grapplemap, vicos
from ._common import project_root

SOURCES = [
    ("GrappleMap", grapplemap.main),
    ("bjjgraph", bjjgraph.main),
    ("ViCoS", lambda: vicos.main([])),
]


def main() -> None:
    racine = project_root()
    print(f"Destination : {racine / 'data' / 'raw'}")
    print("Rappel : rien de ce qui est telecharge ici n'entre dans Git.\n")

    echecs = []
    for nom, fetch in SOURCES:
        try:
            fetch()
        except Exception as exc:  # noqa: BLE001 — on veut continuer malgre un echec
            echecs.append((nom, exc))
            print(f"  ECHEC {nom} : {exc}")
        print("")

    if echecs:
        print("Sources en echec :")
        for nom, exc in echecs:
            print(f"  - {nom} : {exc}")
        raise SystemExit(1)

    print("Les trois sources sont en place. Licences : voir ATTRIBUTIONS.md.")


if __name__ == "__main__":
    main()
