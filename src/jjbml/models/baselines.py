"""Les deux baselines a battre avant de parler de reseau de neurones.

Sans elles, un chiffre ne veut rien dire : 45 % d'accuracy est excellent si la
classe majoritaire en fait 14, et lamentable si une regression logistique en
fait 60.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ClasseMajoritaire:
    """Predit toujours la classe la plus frequente du jeu d'entrainement."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClasseMajoritaire":
        self.classe_ = int(np.bincount(y).argmax())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.classe_, dtype=np.int64)


def regression_logistique() -> object:
    """Modele lineaire sur les 104 caracteristiques, apres standardisation."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=300, C=1.0, n_jobs=-1),
    )
