import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

HOUSES = "Hogwarts House"
LESSONS = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
COLORS = {
    "Gryffindor": "#AA0000",
    "Slytherin": "#1AAA2A",
    "Ravenclaw": "#0E1AAA",
    "Hufflepuff": "#ECB939"
}

def pair_plot(df):
    if HOUSES not in df.columns:
        print(f"Erreur : La colonne '{HOUSES}' est absente du csv.")
        return

    courses = [c for c in df.select_dtypes(include="number").columns if c in (LESSONS)]
    if len(courses) != len(LESSONS):
        print("Erreur : Matières manquantes.")
        return

    sb.set_theme(style="ticks", font_scale=0.7)

    chart = sb.pairplot(
        df[[HOUSES] + courses].dropna(), 
        hue=HOUSES, 
        palette=COLORS,
        corner=True,
        plot_kws={"alpha": 0.5, "s": 5},
        diag_kws={"linewidth": 0.5}
    )

    for ax in chart.axes.flatten():
        if ax is not None:
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_horizontalalignment("right")

    plt.tight_layout()
    plt.subplots_adjust(
        left=0.1,
        bottom=0.05,
    )
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 pair_plot.py <dataset_train.csv>")
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

    pair_plot(df)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        plt.close('all')
        sys.exit(130)
