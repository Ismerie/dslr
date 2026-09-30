#!/usr/bin/env python3
"""Affiche pour chaque feature numérique du dataset : count, mean, std, min, 25%, 50%, 75%, max et calculés sans fonction statistique native."""

import sys
import pandas as pd


def count(values):
    """Compte le nombre de valeurs non manquantes."""
    total = 0
    for x in values:
        if x == x:              # une valeur NaN n'est jamais égale à elle-même
            total += 1
    return total


def mean(values, count_val):
    """Calcule la moyenne."""
    total = sum(values)
    return total / count_val



def std(values, count_val, mean_val):
    total = 0
    for x in values:
        if x == x:
            total += (x - mean_val) ** 2
    return (total / (count_val - 1)) ** 0.5

def my_min(values):
    smallest = values[0]

    for x in values:
        if x == x and x < smallest:
            smallest = x
    return smallest

def my_max(values):
    highest = values[0]

    for x in values:
        if x == x and x > highest:
            highest = x
    return highest

def quantile(values, q):
    valid = [x for x in values if x == x]  # enlève les NaN
    valid.sort()
    n = len(valid)

    idx = q * (n - 1)
    lower = int(idx)               # partie entière (index inférieur)
    upper = min(lower + 1, n - 1)  # index suivant, sans dépasser la fin
    frac = idx - lower              # partie décimale, pour l'interpolation

    return valid[lower] + (valid[upper] - valid[lower]) * frac


def print_table(stats, col_width=20, label_width=8, cols_per_block=3):
    stat_names = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    columns = list(stats.keys())

    for i in range(0, len(columns), cols_per_block):
        block = columns[i:i + cols_per_block]

        # ligne d'en-tête avec les noms de colonnes
        print("".ljust(label_width), end="")
        for col in block:
            print(col[:col_width - 1].rjust(col_width), end="")
        print()

        # une ligne par statistique
        for stat in stat_names:
            print(stat.ljust(label_width), end="")
            for col in block:
                value = stats[col][stat]
                if stat == "Count":
                    print(f"{value:{col_width}.0f}", end="")
                else:
                    print(f"{value:{col_width}.6f}", end="")
            print()
        print()  # ligne vide entre chaque bloc


def describe(df):
    """Construit et affiche le tableau de statistiques."""
    numeric_cols = df.select_dtypes(include="number").columns
    numeric_cols = [col for col in numeric_cols if col not in ("Index", "Hogwarts House")]
    stats = {}

    for col in numeric_cols:
        values = df[col].dropna().tolist()
        values = df[col].dropna().tolist()
        values = df[col].dropna().tolist()
        count_val = count(values)
        mean_val = mean(values, count_val)
        std_val = std(values, count_val, mean_val)

        stats[col] = {
            "Count": count_val,
            "Mean": mean_val,
            "Std": std_val,
            "Min": my_min(values),
            "25%": quantile(values, 0.25),
            "50%": quantile(values, 0.5),
            "75%": quantile(values, 0.75),
            "Max": my_max(values),
        }

    print_table(stats)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 describe.py <dataset.csv>")
        sys.exit(1)

    try:
        df = pd.read_csv(sys.argv[1])
    except FileNotFoundError:
        print(f"Erreur : fichier '{sys.argv[1]}' introuvable.")
        sys.exit(1)

    describe(df)


if __name__ == "__main__":
    main()