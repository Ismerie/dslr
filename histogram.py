import sys
import pandas as pd
import matplotlib.pyplot as plt

LESSONS = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']

def histogram(df):
    courses = [c for c in df.select_dtypes(include="number").columns if c in (LESSONS)]
    houses = df["Hogwarts House"].dropna().unique()

    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    axes = axes.flatten()

    for i, course in enumerate(courses):
        ax = axes[i]
        for house in houses:
            values = df[df["Hogwarts House"] == house][course].dropna()
            ax.hist(values, alpha=0.5, label=house, bins=20)
        ax.set_title(course, fontsize=9)
        ax.set_xlabel("Note")
        ax.set_ylabel("Nombre d'élèves")
        ax.legend(fontsize=6)

    for j in range(len(courses), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 histogram.py <dataset_train.csv>")
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

    histogram(df)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        plt.close('all')
        sys.exit(130)
