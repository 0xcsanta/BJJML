"""Tests du jeu de donnees et des protocoles d'evaluation.

Le test le plus important est `test_le_protocole_par_vue_ne_laisse_aucune_classe_hors_train` :
c'est la propriete qui rend le protocole utilisable, et c'est exactement celle
que le leave-one-roll-out viole.
"""

from __future__ import annotations

import numpy as np
import pytest

from jjbml.data import vicos as dv
from jjbml.parsers.vicos import NB_RANGS, RANG_DE_VUE, ROLLS


def _jeu_factice(n: int = 600, graine: int = 0) -> dv.Jeu:
    """Un jeu synthetique, pour tester la mecanique sans les 199 Mo."""
    rng = np.random.default_rng(graine)
    X = rng.normal(400, 80, (n, dv.N_KEYPOINTS)).astype(np.float32)
    X[:, 2::3] = rng.uniform(0.3, 1.0, X[:, 2::3].shape)  # confiances
    present = np.ones((n, 2), dtype=bool)
    present[: n // 5, 1] = False  # un cinquieme sans pose2
    X[: n // 5, dv.N_PAR_ATHLETE :] = 0.0
    vues = [int(v) for vues in ROLLS for v in vues]
    vue = rng.choice(vues, n)
    roll = np.array(
        [list(ROLLS).index(next(r for r in ROLLS if f"{v:02d}" in r)) for v in vue]
    )
    return dv.Jeu(
        X=X,
        present=present,
        y18=rng.integers(0, 18, n),
        y10=rng.integers(0, 10, n),
        roll=roll,
        vue=vue,
        frame=np.arange(n),
    )


# --------------------------------------------------------------------------
# Mise en forme
# --------------------------------------------------------------------------


def test_normalisation_produit_104_caracteristiques():
    j = _jeu_factice()
    Xn = j.normalise()
    assert Xn.shape == (len(j), dv.N_FEATURES) == (len(j), 104)


def test_normalisation_invariante_au_cadrage():
    """Decaler et zoomer la camera ne doit rien changer : c'est tout l'interet."""
    j = _jeu_factice()
    a = j.normalise()

    decale = j.X.copy().reshape(-1, 2, 17, 3)
    decale[..., :2] = decale[..., :2] * 2.5 + 137.0  # zoom puis translation
    j2 = dv.Jeu(
        X=decale.reshape(len(j), -1), present=j.present, y18=j.y18, y10=j.y10,
        roll=j.roll, vue=j.vue, frame=j.frame,
    )
    assert np.abs(a[:, :102] - j2.normalise()[:, :102]).max() < 1e-3


def test_une_pose_absente_reste_a_zero():
    j = _jeu_factice()
    Xn = j.normalise()
    manquants = ~j.present[:, 1]
    bloc = Xn[manquants, dv.N_PAR_ATHLETE : dv.N_KEYPOINTS]
    assert np.allclose(bloc, 0.0)
    assert np.allclose(Xn[manquants, -1], 0.0)   # drapeau de presence
    assert np.allclose(Xn[manquants, -2], 1.0)


def test_echange_des_athletes_est_involutif():
    j = _jeu_factice()
    Xn = j.normalise()
    Xs, ys = j.echange_athletes(Xn)
    j2 = dv.Jeu(j.X, j.present, ys, j.y10, j.roll, j.vue, j.frame)
    Xss, yss = j2.echange_athletes(Xs)
    assert np.allclose(Xss, Xn)
    assert (yss == j.y18).all()


def test_echange_permute_bien_les_etiquettes():
    assert dv.CLASSES_18[dv._ECHANGE_18[dv.CLASSES_18.index("mount1")]] == "mount2"
    assert dv.CLASSES_18[dv._ECHANGE_18[dv.CLASSES_18.index("back2")]] == "back1"
    # les classes sans role ne bougent pas
    for sans_role in ("standing", "5050_guard"):
        i = dv.CLASSES_18.index(sans_role)
        assert dv._ECHANGE_18[i] == i


# --------------------------------------------------------------------------
# Protocoles
# --------------------------------------------------------------------------


def test_rang_de_vue_couvre_les_16_vues():
    assert len(RANG_DE_VUE) == 16
    assert NB_RANGS == 3
    assert RANG_DE_VUE["00"] == 0 and RANG_DE_VUE["02"] == 2
    assert RANG_DE_VUE["09"] == 0 and RANG_DE_VUE["10"] == 1


def test_les_plis_par_vue_sont_disjoints():
    j = _jeu_factice()
    plis = list(j.plis_par_vue())
    assert len(plis) == NB_RANGS
    tests = [t for _, _, t in plis]
    assert (sum(t.astype(int) for t in tests) == 1).all()  # partition exacte
    for _, train, test in plis:
        assert not (train & test).any()


@pytest.mark.skipif(not dv.chemin_npz().exists(), reason="jeu non prepare")
def test_le_protocole_par_vue_ne_laisse_aucune_classe_hors_train():
    """La propriete qui rend ce protocole utilisable."""
    jeu = dv.charger()
    for _, train, test in jeu.plis_par_vue():
        assert jeu.classes_absentes_du_train(train, test) == []


@pytest.mark.skipif(not dv.chemin_npz().exists(), reason="jeu non prepare")
def test_le_protocole_par_roll_est_bien_impossible():
    """Documente le defaut du jeu de donnees, pour qu'il ne soit pas reintroduit.

    Chaque roll est consacre a quelques positions : en ecarter un retire ses
    classes de l'entrainement. Si ce test venait a echouer, c'est que le jeu de
    donnees a change et qu'il faudrait reconsiderer le protocole.
    """
    jeu = dv.charger()
    manquantes = {
        c
        for _, train, test in jeu.plis_par_roll()
        for c in jeu.classes_absentes_du_train(train, test)
    }
    assert len(manquantes) >= 10


@pytest.mark.skipif(not dv.chemin_npz().exists(), reason="jeu non prepare")
def test_chaque_roll_ne_couvre_que_quelques_classes():
    jeu = dv.charger()
    couverture = jeu.couverture_par_roll()
    assert len(couverture) == 6
    assert all(len(c) <= 9 for c in couverture.values())
    assert min(len(c) for c in couverture.values()) <= 3  # le roll du dos
