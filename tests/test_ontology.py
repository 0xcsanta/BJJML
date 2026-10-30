"""Tests de l'ontologie.

Le test central est `test_critere_de_fin_de_phase_1` : une position nommee dans
l'une des trois sources doit se resoudre dans les deux autres. C'est le critere
d'acceptation ecrit dans PLAN.md.
"""

from __future__ import annotations

import pytest
import yaml

from jjbml import ontology
from jjbml.parsers import grapplemap as gm
from jjbml.parsers import vicos


@pytest.fixture(scope="module")
def onto():
    return ontology.load()


# --------------------------------------------------------------------------
# Chargement et coherence
# --------------------------------------------------------------------------


def test_chargement(onto):
    assert len(onto.positions) == 10
    assert set(onto.positions) == set(onto.correspondances)


def test_les_18_classes_vicos_sont_toutes_mappees(onto):
    for classe in vicos.CLASSES:
        assert onto.resoudre(classe, "vicos") is not None, classe


def test_les_valeurs_sont_dans_zero_un(onto):
    for cle, position in onto.positions.items():
        for role in position.roles:
            v = onto.valeur(cle, role)
            assert 0.0 <= v <= 1.0


def test_la_hierarchie_est_encore_un_brouillon(onto):
    # Garde-fou : ce test tombera quand tu auras relu hierarchy.yaml et mis
    # etat: valide. C'est voulu, ca force a ne pas oublier ce fichier.
    assert onto.est_brouillon


# --------------------------------------------------------------------------
# Resolution entre sources
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "nom, attendu",
    [
        ("mount1", "mount"),
        ("back2", "back"),
        ("5050_guard", "5050_guard"),
        ("side-control", "side_control"),
        ("north-south", "side_control"),      # ViCoS ne les distingue pas
        ("knee-on-belly", "side_control"),
        ("deep-half-guard", "half_guard"),
        ("north_south", "side_control"),
        ("butterfly", "open_guard"),
        ("seatbelt", "back"),
        ("rubber_guard", "closed_guard"),
    ],
)
def test_resolution(onto, nom, attendu):
    assert onto.resoudre(nom) == attendu


def test_resolution_insensible_a_la_casse(onto):
    assert onto.resoudre("Side-Control") == "side_control"


def test_nom_inconnu(onto):
    assert onto.resoudre("berimbolo-des-familles") is None


def test_traduction_entre_sources(onto):
    assert "mount" in onto.traduire("mount1", "bjjgraph")
    assert "mount1" in onto.traduire("s-mount", "vicos")
    assert "turtle" in onto.traduire("turtle2", "grapplemap")


def test_critere_de_fin_de_phase_1(onto):
    """Toute position nommee dans une source se resout dans les deux autres."""
    for cle in onto.positions:
        for source in ontology.SOURCES:
            for nom in onto.vers(cle, source):
                assert onto.resoudre(nom, source) == cle
                for autre in ontology.SOURCES:
                    if autre == source:
                        continue
                    traduit = onto.traduire(nom, autre)
                    if autre == "bjjgraph" and cle == "takedown":
                        # Seule exception, documentee : bjjgraph modelise les
                        # takedowns comme des transitions, pas des positions.
                        assert traduit == []
                    else:
                        assert traduit, f"{source}:{nom} ne se traduit pas en {autre}"


# --------------------------------------------------------------------------
# La validation doit mordre
# --------------------------------------------------------------------------


def test_un_nom_range_sous_deux_positions_est_refuse(tmp_path):
    source = ontology.dossier_ontologie()
    for f in ("positions.yaml", "mapping.yaml", "hierarchy.yaml"):
        (tmp_path / f).write_text((source / f).read_text(encoding="utf-8"), encoding="utf-8")

    mapping = yaml.safe_load((tmp_path / "mapping.yaml").read_text(encoding="utf-8"))
    mapping["correspondances"]["mount"]["grapplemap"].append("turtle")  # deja sous turtle
    (tmp_path / "mapping.yaml").write_text(yaml.safe_dump(mapping), encoding="utf-8")

    with pytest.raises(ontology.OntologieInvalide, match="turtle"):
        ontology.load(tmp_path)


def test_une_valeur_hors_bornes_est_refusee(tmp_path):
    source = ontology.dossier_ontologie()
    for f in ("positions.yaml", "mapping.yaml", "hierarchy.yaml"):
        (tmp_path / f).write_text((source / f).read_text(encoding="utf-8"), encoding="utf-8")

    hier = yaml.safe_load((tmp_path / "hierarchy.yaml").read_text(encoding="utf-8"))
    hier["valeurs"]["mount"]["dessus"] = 1.4
    (tmp_path / "hierarchy.yaml").write_text(yaml.safe_dump(hier), encoding="utf-8")

    with pytest.raises(ontology.OntologieInvalide, match=r"hors de"):
        ontology.load(tmp_path)


# --------------------------------------------------------------------------
# Integration : les noms de la table existent-ils vraiment dans les sources ?
# C'est ce qui attrape les fautes de frappe, que la coherence interne ne voit pas.
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not gm.chemin_par_defaut().exists(), reason="GrappleMap.txt non telecharge"
)
def test_tous_les_tags_grapplemap_existent_vraiment(onto):
    reels = set()
    for ligne in gm.chemin_par_defaut().read_text(encoding="utf-8", errors="replace").splitlines():
        if ligne.startswith("tags:"):
            reels.update(ligne[5:].split())

    inconnus = [
        tag
        for cle in onto.positions
        for tag in onto.vers(cle, "grapplemap")
        if tag not in reels
    ]
    assert not inconnus, f"tags absents de GrappleMap : {inconnus}"


@pytest.mark.skipif(
    not __import__("jjbml.parsers.bjjgraph", fromlist=["x"]).chemin_par_defaut().exists(),
    reason="graph.json.gz non telecharge",
)
def test_tous_les_hubs_bjjgraph_existent_vraiment(onto):
    from jjbml.parsers import bjjgraph

    reels = set(bjjgraph.hubs())
    inconnus = [
        hub
        for cle in onto.positions
        for hub in onto.vers(cle, "bjjgraph")
        if hub not in reels
    ]
    assert not inconnus, f"hubs absents de bjjgraph : {inconnus}"
