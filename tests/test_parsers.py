"""Tests des parsers.

Les tests unitaires n'ont besoin d'aucune donnee telechargee. Les tests
d'integration sont ignores automatiquement si data/ est vide, pour qu'un clone
frais passe au vert sans avoir rien recupere.
"""

from __future__ import annotations

import numpy as np
import pytest

from jjbml.parsers import grapplemap as gm
from jjbml.parsers import vicos


# --------------------------------------------------------------------------
# Encodage base62
# --------------------------------------------------------------------------


def encode_valeur(v: float) -> str:
    """Inverse de decode : reproduit l'encodeur de persistence.cpp."""
    i = round(v * 1000)
    return gm.B62[i // 62] + gm.B62[i % 62]


def test_alphabet_base62():
    assert len(gm.B62) == 62
    assert gm.B62[0] == "a" and gm.B62[25] == "z"
    assert gm.B62[26] == "A" and gm.B62[51] == "Z"
    assert gm.B62[52] == "0" and gm.B62[61] == "9"


def test_decode_keyframe_aller_retour():
    rng = np.random.default_rng(0)
    pose = np.stack(
        [
            rng.uniform(-2, 1.8, gm.JOINTS_TOTAL),  # x
            rng.uniform(0, 3.8, gm.JOINTS_TOTAL),  # y
            rng.uniform(-2, 1.8, gm.JOINTS_TOTAL),  # z
        ],
        axis=1,
    )
    brut = pose.copy()
    brut[:, 0] += 2.0
    brut[:, 2] += 2.0
    chars = "".join(encode_valeur(v) for v in brut.reshape(-1))

    decode = gm.decode_keyframe(chars)
    assert decode.shape == (gm.JOINTS_TOTAL, 3)
    # l'encodage quantifie au millimetre
    assert np.abs(decode - pose).max() < 1e-3


def test_decode_keyframe_refuse_une_mauvaise_longueur():
    with pytest.raises(ValueError):
        gm.decode_keyframe("ab" * 10)


# --------------------------------------------------------------------------
# Signature invariante
# --------------------------------------------------------------------------


def test_signature_invariante_par_rotation_et_translation():
    """C'est la propriete sur laquelle repose tout le raccordement du graphe."""
    rng = np.random.default_rng(1)
    pose = rng.uniform(-1, 1, (gm.JOINTS_TOTAL, 3))

    angle = 0.7
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    bougee = pose @ rotation.T + np.array([3.0, 0.0, -1.5])

    assert np.abs(gm.signature(pose) - gm.signature(bougee)).max() < 1e-9


def test_signature_distingue_deux_poses_differentes():
    rng = np.random.default_rng(2)
    a = rng.uniform(-1, 1, (gm.JOINTS_TOTAL, 3))
    b = rng.uniform(-1, 1, (gm.JOINTS_TOTAL, 3))
    assert np.abs(gm.signature(a) - gm.signature(b)).max() > gm.TOLERANCE_APPARIEMENT


# --------------------------------------------------------------------------
# Lecture du format
# --------------------------------------------------------------------------


def _keyframe_factice(graine: int) -> list[str]:
    rng = np.random.default_rng(graine)
    chars = "".join(
        encode_valeur(v) for v in rng.uniform(0, 3.8, gm.VALEURS_PAR_KEYFRAME)
    )
    return [chars[i : i + 69] for i in range(0, len(chars), 69)]


def test_lecture_des_entrees():
    kf1, kf2 = _keyframe_factice(10), _keyframe_factice(11)
    texte = "\n".join(
        [
            "mount\\nhaute",
            "tags: mount top_kneeling",
            *[f"    {l}" for l in kf1],
            "bottom bridges",
            "properties: top",
            "ref: Quelqu'un, Un instructional @ 3:21",
            *[f"    {l}" for l in kf1],
            *[f"    {l}" for l in kf2],
            "entree sans pose",
            "todo: a dessiner",
        ]
    )
    entrees = gm._lire_entrees(texte)
    assert len(entrees) == 3

    position, transition, vide = entrees
    assert position.nom_lisible == "mount haute"
    assert position.tags == ["mount", "top_kneeling"]
    assert position.est_position and not position.est_transition

    assert transition.est_transition
    assert len(transition.keyframes) == 2
    assert transition.properties == ["top"]
    assert len(transition.refs) == 1

    assert vide.keyframes is None


# --------------------------------------------------------------------------
# ViCoS : le groupe de split
# --------------------------------------------------------------------------


def test_les_18_classes_couvrent_10_positions():
    assert len(vicos.CLASSES) == 18
    assert len(vicos.POSITIONS) == 10


def test_seize_vues_pour_six_rolls():
    """Le point qui sauve la phase 2 : 16 vues, mais seulement 6 combats."""
    assert len(vicos.VUE_VERS_ROLL) == 16
    assert len(vicos.ROLLS) == 6


def test_deux_vues_d_un_meme_roll_tombent_dans_le_meme_groupe():
    # 00, 01 et 02 sont le meme combat filme par trois cameras : les separer
    # entre entrainement et test ferait fuiter le meme instant.
    assert vicos.roll("0000042") == vicos.roll("0100042") == vicos.roll("0200042")
    # alors que 03 est un autre combat
    assert vicos.roll("0300042") != vicos.roll("0000042")


def test_decoupage_de_l_identifiant():
    assert vicos.vue("1200345") == "12"
    assert vicos.numero_de_frame("1200345") == 345


# --------------------------------------------------------------------------
# Integration : seulement si les donnees sont la
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not gm.chemin_par_defaut().exists(), reason="GrappleMap.txt non telecharge"
)
def test_integration_graphe_sans_arete_pendante():
    g = gm.load()
    assert len(g.positions) > 500
    assert len(g.transitions) > 1000
    assert len(g.aretes) == len(g.transitions)
    # toute arete pointe vers un noeud existant : le graphe est complet
    for a in g.aretes:
        assert 0 <= a.depart < len(g.noeuds)
        assert 0 <= a.arrivee < len(g.noeuds)
