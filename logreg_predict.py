#!/usr/bin/env python3
"""Prédit la maison de chaque élève avec les poids appris par logreg_train.py et écrit houses.csv."""

import sys
import json
import numpy as np
import pandas as pd
from logreg_train import sigmoid

HOUSES = "Hogwarts House"
OUTPUT_FILE = "houses.csv"


def load_model(path):
    """Lit le fichier de poids et renvoie les matières, moyennes, écarts-types et thetas."""
    try:
        with open(path) as f:
            model = json.load(f)
    except FileNotFoundError:
        print(f"Erreur : le fichier '{path}' est introuvable.")
        sys.exit(1)
    except IsADirectoryError:
        print(f"Erreur : '{path}' est un dossier, pas un fichier de poids.")
        sys.exit(1)
    except PermissionError:
        print(f"Erreur : permission refusée pour lire '{path}'.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Erreur : '{path}' n'est pas un fichier JSON valide.")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur inattendue lors de la lecture de '{path}' : {e}")
        sys.exit(1)

    expected_keys = {"features", "means", "stds", "thetas"}
    model_keys = set(model.keys())
    missing = expected_keys - model_keys
    if missing:
        print(f"Erreur : clé(s) manquante(s) dans '{path}' : {', '.join(missing)}")
        sys.exit(1)
    extra = model_keys - expected_keys
    if extra:
        print(f"Erreur : clé(s) inconnue(s) ou inattendue(s) dans '{path}' : {', '.join(extra)}")
        sys.exit(1)

    features = model["features"]
    means = np.array(model["means"])
    stds = np.array(model["stds"])
    houses = list(model["thetas"].keys())
    # une ligne par maison, une colonne par poids : forme (4, 10)
    thetas = np.array([model["thetas"][house] for house in houses])
    return features, means, stds, houses, thetas


def load_dataset(path):
    """Lit le CSV à prédire, avec la même gestion d'erreurs que les autres scripts."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Erreur : le fichier '{path}' est introuvable.")
    except IsADirectoryError:
        print(f"Erreur : '{path}' est un dossier, pas un fichier CSV.")
    except PermissionError:
        print(f"Erreur : permission refusée pour lire '{path}'.")
    except pd.errors.EmptyDataError:
        print(f"Erreur : le fichier '{path}' est vide.")
    except Exception as e:
        print(f"Erreur inattendue lors de la lecture de '{path}' : {e}")
    sys.exit(1)


def prepare_data(df, features, means, stds):
    """Prépare les notes exactement comme au train, sans supprimer d'élève."""
    raw = df[features].to_numpy(dtype=float)

    # Cases vides : on les remplace par la moyenne du train de la matière
    raw = np.where(np.isnan(raw), means, raw)

    # Normalisation avec les moyennes et écarts-types du train
    normalized = (raw - means) / stds

    # Colonne de 1 au début, pour le biais theta0
    return np.column_stack([np.ones(len(normalized)), normalized])


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 logreg_predict.py <dataset_test.csv> <weights.json>")
        sys.exit(1)

    features, means, stds, houses, thetas = load_model(sys.argv[2])
    df = load_dataset(sys.argv[1])
    missing = [col for col in features if col not in df.columns]
    if missing:
        print(f"Erreur : colonnes manquantes dans le csv : {', '.join(missing)}")
        sys.exit(1)

    X = prepare_data(df, features, means, stds)

    # Probabilité de chaque maison pour chaque élève : forme (nb élèves, 4)
    probas = sigmoid(X @ thetas.T)

    # Pour chaque élève, la maison avec la plus grande probabilité
    best = np.argmax(probas, axis=1)
    predictions = [houses[i] for i in best]

    index = df["Index"] if "Index" in df.columns else range(len(df))
    result = pd.DataFrame({"Index": index, HOUSES: predictions})
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Prédictions écrites dans {OUTPUT_FILE}, ({len(result)} élèves)")


if __name__ == "__main__":
    main()
