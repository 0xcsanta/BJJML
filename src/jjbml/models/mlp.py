"""Le premier reseau du projet : un perceptron multicouche sur 104 entrees.

Rien d'exotique, et c'est voulu. Si un reseau ne bat pas une regression
logistique sur des caracteristiques aussi directes, ce n'est pas qu'il manque
de couches : c'est que le probleme est ailleurs. Ici il est dans les donnees.

Deux details qui comptent plus que l'architecture :

- **Les poids de classe.** Les classes vont de 2 586 a 16 722 exemples. Sans
  ponderation, le reseau optimise l'accuracy brute et neglige les classes
  rares, alors qu'on mesure l'accuracy equilibree. On pondere donc la perte par
  l'inverse de la frequence.
- **L'arret sur validation.** Une part du jeu d'entrainement est mise de cote
  pour garder l'epoque qui generalise le mieux, plutot que la derniere. Sans
  ca, on choisit le modele qui a le mieux memorise.

Tout tourne sur CPU : 104 entrees, quelques dizaines de milliers d'exemples,
c'est l'affaire de quelques dizaines de secondes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MLP:
    """Interface fit/predict, pour etre interchangeable avec les baselines."""

    def __init__(
        self,
        couches: tuple[int, ...] = (256, 128),
        epoques: int = 40,
        lot: int = 512,
        lr: float = 1e-3,
        dropout: float = 0.2,
        part_validation: float = 0.15,
        graine: int = 0,
        verbeux: bool = False,
    ):
        self.couches = couches
        self.epoques = epoques
        self.lot = lot
        self.lr = lr
        self.dropout = dropout
        self.part_validation = part_validation
        self.graine = graine
        self.verbeux = verbeux

    def _construire(self, n_entrees: int, n_classes: int) -> nn.Module:
        blocs: list[nn.Module] = []
        precedent = n_entrees
        for taille in self.couches:
            blocs += [nn.Linear(precedent, taille), nn.ReLU(), nn.Dropout(self.dropout)]
            precedent = taille
        blocs.append(nn.Linear(precedent, n_classes))
        return nn.Sequential(*blocs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLP":
        torch.manual_seed(self.graine)
        rng = np.random.default_rng(self.graine)

        n_classes = int(y.max()) + 1
        self.moyenne_ = X.mean(axis=0, keepdims=True)
        self.ecart_ = np.clip(X.std(axis=0, keepdims=True), 1e-6, None)
        Xn = (X - self.moyenne_) / self.ecart_

        # validation prelevee au hasard : les plis sont deja separes par camera,
        # donc l'enjeu ici est seulement de choisir l'epoque d'arret.
        melange = rng.permutation(len(Xn))
        n_val = int(len(Xn) * self.part_validation)
        i_val, i_tr = melange[:n_val], melange[n_val:]

        Xtr = torch.tensor(Xn[i_tr], dtype=torch.float32)
        ytr = torch.tensor(y[i_tr], dtype=torch.long)
        Xva = torch.tensor(Xn[i_val], dtype=torch.float32)
        yva = torch.tensor(y[i_val], dtype=torch.long)

        effectifs = np.bincount(y[i_tr], minlength=n_classes).astype(np.float64)
        poids = np.where(effectifs > 0, len(i_tr) / np.clip(effectifs, 1, None), 0.0)
        poids = poids / poids[poids > 0].mean()

        self.reseau_ = self._construire(X.shape[1], n_classes)
        perte = nn.CrossEntropyLoss(weight=torch.tensor(poids, dtype=torch.float32))
        optim = torch.optim.Adam(self.reseau_.parameters(), lr=self.lr)

        meilleur, meilleurs_poids = -1.0, None
        for epoque in range(self.epoques):
            self.reseau_.train()
            ordre = torch.randperm(len(Xtr))
            for debut in range(0, len(Xtr), self.lot):
                lot = ordre[debut : debut + self.lot]
                optim.zero_grad()
                perte(self.reseau_(Xtr[lot]), ytr[lot]).backward()
                optim.step()

            self.reseau_.eval()
            with torch.no_grad():
                pred = self.reseau_(Xva).argmax(dim=1)
            # rappel moyen par classe, la metrique qu'on vise vraiment
            scores = [
                float((pred[yva == c] == c).float().mean())
                for c in range(n_classes)
                if (yva == c).any()
            ]
            score = float(np.mean(scores))
            if score > meilleur:
                meilleur = score
                meilleurs_poids = {
                    k: v.clone() for k, v in self.reseau_.state_dict().items()
                }
            if self.verbeux and epoque % 10 == 0:
                print(f"    epoque {epoque:3d}  validation {score:.3f}")

        if meilleurs_poids is not None:
            self.reseau_.load_state_dict(meilleurs_poids)
        self.score_validation_ = meilleur
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.reseau_.eval()
        Xn = torch.tensor(
            (X - self.moyenne_) / self.ecart_, dtype=torch.float32
        )
        with torch.no_grad():
            sorties = []
            for debut in range(0, len(Xn), 4096):
                sorties.append(self.reseau_(Xn[debut : debut + 4096]).argmax(dim=1))
        return torch.cat(sorties).numpy()
