"""Briques communes aux scripts de telechargement.

Aucune dependance externe : la bibliotheque standard suffit, ce qui permet de
recuperer les sources avant meme d'avoir installe quoi que ce soit.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

USER_AGENT = "jjb-ML/0.1 (projet d'apprentissage, non commercial)"
CHUNK = 1 << 20  # 1 Mo


def project_root() -> Path:
    """Racine du depot, deduite de l'emplacement de ce fichier."""
    return Path(__file__).resolve().parents[3]


def raw_dir(source: str) -> Path:
    """Dossier d'accueil d'une source, cree au besoin. Toujours sous data/raw/."""
    d = project_root() / "data" / "raw" / source
    d.mkdir(parents=True, exist_ok=True)
    return d


def human(n: float) -> str:
    """Taille lisible : 199.1 Mo plutot que 208848730."""
    for unit in ("o", "Ko", "Mo", "Go"):
        if n < 1024 or unit == "Go":
            return f"{n:.0f} {unit}" if unit == "o" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} Go"


def urlopen(url: str, *, method: str = "GET"):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req)


def remote_size(url: str) -> int | None:
    """Taille annoncee par le serveur, ou None s'il ne la donne pas."""
    try:
        with urlopen(url, method="HEAD") as r:
            length = r.headers.get("Content-Length")
    except OSError:
        return None
    return int(length) if length else None


def download(url: str, dest: Path, *, expected_size: int | None = None) -> Path:
    """Telecharge url vers dest, avec progression.

    Idempotent : un fichier deja present et de la bonne taille n'est pas
    retelecharge. L'ecriture passe par un .part renomme a la fin, pour qu'une
    interruption ne laisse jamais un fichier tronque qui passerait pour complet.
    """
    if expected_size is None:
        expected_size = remote_size(url)

    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        print(f"  deja present : {dest.name} ({human(expected_size)})")
        return dest

    if expected_size:
        print(f"  telechargement : {dest.name} ({human(expected_size)})")
    else:
        print(f"  telechargement : {dest.name}")

    tmp = dest.with_name(dest.name + ".part")
    done = 0
    with urlopen(url) as r, tmp.open("wb") as out:
        while True:
            chunk = r.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if expected_size:
                pct = 100 * done / expected_size
                print(f"\r    {human(done)} / {human(expected_size)}  ({pct:.0f} %)",
                      end="", file=sys.stderr, flush=True)
            else:
                print(f"\r    {human(done)}", end="", file=sys.stderr, flush=True)
    print("", file=sys.stderr)

    # Une connexion coupee en cours de route se manifeste par un read() vide,
    # pas par une exception : sans ce controle, un fichier tronque serait
    # renomme et passerait ensuite pour complet.
    if expected_size and done != expected_size:
        tmp.unlink(missing_ok=True)
        raise OSError(
            f"telechargement incomplet : {human(done)} recus sur {human(expected_size)} "
            "attendus. Rien n'a ete conserve, relancer la commande."
        )

    tmp.replace(dest)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def record(source: str, *, licence: str, origine: str, fichiers: list[Path],
           note: str = "") -> Path:
    """Ecrit data/raw/<source>/SOURCE.json : provenance, licence, empreintes.

    C'est la trace de conformite : a tout moment on sait d'ou vient chaque
    octet present sur le disque, et sous quelle licence il s'y trouve.
    """
    d = raw_dir(source)
    payload = {
        "source": source,
        "origine": origine,
        "licence": licence,
        "telecharge_le": date.today().isoformat(),
        "note": note,
        "fichiers": [
            {"nom": p.name, "octets": p.stat().st_size, "sha256": sha256(p)}
            for p in fichiers
            if p.exists()
        ],
    }
    path = d / "SOURCE.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  provenance : {path.relative_to(project_root())}")
    return path
