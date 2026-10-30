"""bjjgraph — ontologie des positions, transitions et soumissions.

https://github.com/diogoseca/bjjgraph — PolyForm Noncommercial 1.0.0

On recupere l'artefact publie `graph.json.gz` (environ 6,5 Mo) de la derniere
release, plutot que de cloner les ~300 Mo du depot.

ATTENTION, et c'est le point le plus important de ce fichier : les
probabilites de cette source sont des « authored estimates », ecrites a la
main et jamais mesurees. Son propre README precise qu'elles attendent une revue
par un panel de ceintures noires qui n'a pas eu lieu.

On en prend donc **la structure du graphe, et rien d'autre**. Les probabilites
servent uniquement de point de comparaison pour nos mesures (PLAN.md, phase 6),
jamais de verite terrain.
"""

from __future__ import annotations

import json

from ._common import download, raw_dir, record, urlopen

API_RELEASE = "https://api.github.com/repos/diogoseca/bjjgraph/releases/latest"
DEPOT = "https://github.com/diogoseca/bjjgraph"
ASSET = "graph.json.gz"
LICENCE = "PolyForm Noncommercial 1.0.0"


def main() -> None:
    print("bjjgraph - PolyForm Noncommercial 1.0.0")
    with urlopen(API_RELEASE) as r:
        release = json.load(r)

    tag = release.get("tag_name", "inconnue")
    asset = next((a for a in release.get("assets", []) if a.get("name") == ASSET), None)
    if asset is None:
        raise SystemExit(
            f"  l'artefact {ASSET} est absent de la release {tag}. "
            f"Verifier {DEPOT}/releases"
        )

    print(f"  release {tag}")
    dest = raw_dir("bjjgraph") / ASSET
    download(asset["browser_download_url"], dest, expected_size=asset.get("size"))
    record(
        "bjjgraph",
        licence=LICENCE,
        origine=f"{DEPOT} (release {tag}, actif {ASSET})",
        fichiers=[dest],
        note=(
            "STRUCTURE UNIQUEMENT. Les probabilites de cette source sont des estimations "
            "ecrites a la main, jamais mesurees : ne jamais les utiliser comme verite terrain."
        ),
    )


if __name__ == "__main__":
    main()
