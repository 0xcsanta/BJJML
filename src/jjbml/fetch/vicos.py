"""ViCoS Brazilian Jiu-Jitsu Positions Dataset.

https://vicos.si/resources/jiujitsu/ — CC BY-NC-SA 4.0

Deux fichiers, de tailles tres differentes :

  annotations.json    199 Mo    labels + keypoints 2D COCO des deux athletes
  images.zip          9,35 Go   les 120 279 images

Par defaut on ne prend que les annotations, et c'est volontaire : les keypoints
y sont deja, donc le premier modele (PLAN.md, phase 2) s'entraine sans avoir
besoin d'une seule image. Les 9,35 Go ne deviennent utiles qu'a partir du moment
ou l'on veut entrainer la perception sur les pixels.

Citation exigee par la licence :
  Hudovernik, V. et Skocaj, D. — Video-Based Detection of Combat Positions and
  Automatic Scoring in Jiu-jitsu. MMSports '22, Lisbonne, 2022, p. 55-63.
"""

from __future__ import annotations

import argparse

from ._common import download, human, raw_dir, record

BASE = "https://data.vicos.si/datasets/JuiJuitsu"
ANNOTATIONS = f"{BASE}/annotations.json"
IMAGES = f"{BASE}/images.zip"
PAGE = "https://vicos.si/resources/jiujitsu/"
LICENCE = "CC BY-NC-SA 4.0"
CITATION = (
    "Hudovernik, V. et Skocaj, D. - Video-Based Detection of Combat Positions and "
    "Automatic Scoring in Jiu-jitsu. MMSports '22, Lisbonne, 2022, p. 55-63."
)
TAILLE_IMAGES = 10_043_058_772


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Telecharge le dataset ViCoS.")
    parser.add_argument(
        "--images",
        action="store_true",
        help=f"telecharge aussi images.zip ({human(TAILLE_IMAGES)}), inutile avant la phase 4",
    )
    args = parser.parse_args(argv)

    print("ViCoS BJJ Positions Dataset - CC BY-NC-SA 4.0")
    print(f"  citation : {CITATION}")

    d = raw_dir("vicos")
    fichiers = [download(ANNOTATIONS, d / "annotations.json")]

    if args.images:
        print(f"  images.zip fait {human(TAILLE_IMAGES)} - ca va etre long.")
        fichiers.append(download(IMAGES, d / "images.zip", expected_size=TAILLE_IMAGES))
    else:
        print(f"  images.zip ({human(TAILLE_IMAGES)}) ignore - relancer avec --images si besoin.")

    record(
        "vicos",
        licence=LICENCE,
        origine=PAGE,
        fichiers=fichiers,
        note=(
            f"Citation exigee par la licence : {CITATION} "
            "Partage dans les memes conditions (SA) : toute donnee derivee redistribuee "
            "porte la meme licence."
        ),
    )


if __name__ == "__main__":
    main()
