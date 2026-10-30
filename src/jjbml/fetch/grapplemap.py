"""GrappleMap — graphe de positions et transitions, domaine public.

https://github.com/Eelis/GrappleMap

Environ 2,7 Mo. Fournit l'ossature du graphe (etage B) et surtout un
vocabulaire de tags bien plus fin que les 10 classes de ViCoS : top_underhook,
crossface, collar_tie, bottom_post_hand... soit exactement la notion de grips
qui manque pour conseiller un mouvement.

Le format est ad-hoc et non documente ; ecrire le parser est la phase 1.
"""

from __future__ import annotations

from ._common import download, raw_dir, record

URL = "https://raw.githubusercontent.com/Eelis/GrappleMap/master/GrappleMap.txt"
LICENCE = "Domaine public"


def main() -> None:
    print("GrappleMap - domaine public")
    dest = raw_dir("grapplemap") / "GrappleMap.txt"
    download(URL, dest)
    record(
        "grapplemap",
        licence=LICENCE,
        origine=URL,
        fichiers=[dest],
        note="Base texte ad-hoc, schema non documente (cf. PLAN.md, phase 1).",
    )


if __name__ == "__main__":
    main()
