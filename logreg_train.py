#!/usr/bin/env python3
"""Entraîne un classifieur one-vs-all par régression logistique et sauvegarde les poids."""

import sys
import json
import numpy as np
import pandas as pd
from describe import count, mean, std

HOUSES = "Hogwarts House"
# Matières retirées :
# - Arithmancy, Care of Magical Creatures : répartition homogène entre les maisons (histogram)
# - Defense Against the Dark Arts : corrélation de -1 avec Astronomy, redondante (scatter_plot)
# - Potions : les maisons se chevauchent beaucoup (pair_plot)
FEATURES = [
    'Astronomy',
    'Herbology',
    'Divination',
    'Muggle Studies',
    'Ancient Runes',
    'History of Magic',
    'Transfiguration',
    'Charms',
    'Flying',
]
LEARNING_RATE = 0.1
ITERATIONS = 2000
WEIGHTS_FILE = "weights.json"
EPSILON = 1e-15  # évite log(0) dans le coût


def sigmoid(z):
    """g(z) = 1 / (1 + e^-z), appliquée à chaque élément de z."""
    return 1 / (1 + np.exp(-z))


def cost(h, y):
    """Log loss : -(1/m) * somme(y*log(h) + (1-y)*log(1-h))."""
    m = len(y)
    h = np.clip(h, EPSILON, 1 - EPSILON)  # garde h strictement entre 0 et 1
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m


def gradient(X, h, y):
    """Dérivée du coût par rapport à theta : (1/m) * X^T (h - y)."""
    m = len(y)
    return X.T @ (h - y) / m


def train_one(X, y):
    """Descente de gradient pour UN classifieur binaire. Renvoie theta."""
    theta = np.zeros(X.shape[1])
    for i in range(ITERATIONS):
        h = sigmoid(X @ theta)              # probabilités prédites pour chaque élève
        theta = theta - LEARNING_RATE * gradient(X, h, y)
        if i % 500 == 0:
            print(f"  itération {i:4d} : coût = {cost(h, y):.4f}")
    print(f"  itération {ITERATIONS:4d} : coût = {cost(sigmoid(X @ theta), y):.4f}")
    return theta


def prepare_data(df):
    """Nettoie, normalise et ajoute la colonne de biais.

    Renvoie X (m, n+1), le tableau des maisons, et les stats de normalisation.
    """
    df = df.dropna(subset=FEATURES + [HOUSES])
    raw = df[FEATURES].to_numpy(dtype=float)

    # Moyenne et écart-type de chaque matière (colonne par colonne),
    means = []
    stds = []
    for j in range(len(FEATURES)):
        values = raw[:, j].tolist()      # toutes les notes de la matière j
        count_val = count(values)
        mean_val = mean(values, count_val)
        means.append(mean_val)
        stds.append(std(values, count_val, mean_val))
    means = np.array(means)
    stds = np.array(stds)

    # Normalisation : chaque colonne est centrée sur 0 avec un écart-type de 1
    normalized = (raw - means) / stds

    # Colonne de 1 au début, pour le biais theta0
    X = np.column_stack([np.ones(len(normalized)), normalized])

    return X, df[HOUSES].to_numpy(), means, stds


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 logreg_train.py <dataset_train.csv>")
        sys.exit(1)

    file = sys.argv[1]

    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        print(f"Erreur : le fichier '{file}' est introuvable.")
        sys.exit(1)
    except IsADirectoryError:
        print(f"Erreur : '{file}' est un dossier, pas un fichier CSV.")
        sys.exit(1)
    except PermissionError:
        print(f"Erreur : permission refusée pour lire '{file}'.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Erreur : le fichier '{file}' est vide.")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur inattendue lors de la lecture de '{file}' : {e}")
        sys.exit(1)

    missing = [col for col in FEATURES + [HOUSES] if col not in df.columns]
    if missing:
        print(f"Erreur : colonnes manquantes dans le csv : {', '.join(missing)}")
        sys.exit(1)

    X, houses, means, stds = prepare_data(df)
    if len(X) == 0:
        print("Erreur : aucun élève complet pour l'entraînement.")
        sys.exit(1)

    # One-vs-all : un classifieur binaire par maison
    thetas = {}
    for house in np.unique(houses):
        print(f"Entraînement : {house} ou pas ?")
        y = (houses == house).astype(float)  # 1 si l'élève est dans cette maison, sinon 0
        thetas[house] = train_one(X, y)

    model = {
        "features": FEATURES,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "thetas": {house: theta.tolist() for house, theta in thetas.items()},
    }
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(model, f, indent=2)
    print(f"Poids sauvegardés dans {WEIGHTS_FILE}")


if __name__ == "__main__":
    main()
