import sys
import pandas as pd
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 histogram.py <dataset_train.csv>")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    courses = [c for c in df.select_dtypes(include="number").columns if c != "Index"]
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

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
