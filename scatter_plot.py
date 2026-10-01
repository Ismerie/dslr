import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

LESSONS = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
COLORS = {
    "Gryffindor": "#AA0000",
    "Slytherin": "#1AAA2A",
    "Ravenclaw": "#0E1AAA",
    "Hufflepuff": "#ECB939"
}

def get_corrs(df, courses):
    """Calcule les paires avec la corrélation max (positive) et min (négative)."""
    corr_matrix = df[courses].dropna().corr()    # récupération de la matrice de corrélation
    matrix_np = corr_matrix.to_numpy(copy=True) # création d'une matrice numpy modifiable
    np.fill_diagonal(matrix_np, 0)  # mise à 0 de la diagonale qui correspond à deux mmême matières, donc à exclure
    clean_corr = pd.DataFrame(matrix_np, index=corr_matrix.index, columns=corr_matrix.columns)  # réetour à la matrice de corrélation avec les noms des matières
    
    # récupération de la paire avec la corrélation max et la corrélation max inversée
    feat_x_max, feat_y_max = clean_corr.unstack().idxmax()
    value_max = clean_corr.unstack().max()
    feat_x_min, feat_y_min = clean_corr.unstack().idxmin()
    value_min = clean_corr.unstack().min()

    return (feat_x_max, feat_y_max, value_max), (feat_x_min, feat_y_min, value_min)


def scatter_plot(df):
    courses = [c for c in df.select_dtypes(include="number").columns if c in (LESSONS)]
    houses = df["Hogwarts House"].dropna().unique()

    max_glob, min_glob = get_corrs(df, courses)

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    ax_glob_max = axes[0, 0]
    ax_glob_min = axes[1, 0]

    for idx, house in enumerate(houses, start=1):
        house_df = df[df["Hogwarts House"] == house]
        color_rgba = mcolors.to_rgba(COLORS.get(house), alpha=0.5)

        # Remplissage des graphiques globaux
        data_max = house_df[[max_glob[0], max_glob[1]]].dropna()
        ax_glob_max.scatter(
            data_max[max_glob[0]], 
            data_max[max_glob[1]],  
            label=house,
            color=color_rgba
        )
        data_min = house_df[[min_glob[0], min_glob[1]]].dropna()
        ax_glob_min.scatter(
            data_min[min_glob[0]], 
            data_min[min_glob[1]],  
            label=house,
            color=color_rgba
        )

        # Remplissage des graphiques par maison
        max_h, min_h = get_corrs(house_df, courses)
        ax_h_max = axes[0, idx]
        ax_h_min = axes[1, idx]
        data_max = house_df[[max_h[0], max_h[1]]].dropna()
        data_min = house_df[[min_h[0], min_h[1]]].dropna()
        ax_h_max.scatter(
            data_max[max_h[0]], 
            data_max[max_h[1]],  
            label=house,
            color=color_rgba
        )
        ax_h_max.set_title(f"Corrélation max positive de {house} ({max_h[2]:.2f})")
        ax_h_max.set_xlabel(max_h[0])
        ax_h_max.set_ylabel(max_h[1])
        ax_h_max.legend()
        ax_h_min.scatter(
            data_min[min_h[0]], 
            data_min[min_h[1]],  
            label=house,
            color=color_rgba
        )
        ax_h_min.set_title(f"Corrélation max négative de {house} ({min_h[2]:.2f})")
        ax_h_min.set_xlabel(min_h[0])
        ax_h_min.set_ylabel(min_h[1])
        ax_h_min.legend()

    ax_glob_max.set_title(f"Corrélation max positive ({max_glob[2]:.2f})")
    ax_glob_max.set_xlabel(max_glob[0])
    ax_glob_max.set_ylabel(max_glob[1])
    ax_glob_max.legend()
    ax_glob_min.set_title(f"Corrélation max négative ({max_glob[2]:.2f})")
    ax_glob_min.set_xlabel(min_glob[0])
    ax_glob_min.set_ylabel(min_glob[1])
    ax_glob_min.legend()

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 scatter_plot.py <dataset_train.csv>")
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

    scatter_plot(df)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        plt.close('all')
        sys.exit(130)
