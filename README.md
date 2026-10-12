# DSLR — Data Science × Logistic Regression

Projet 42 : retrouver la maison de Poudlard des élèves à partir de leurs notes, avec une régression logistique **one-vs-all** codée à la main (descente de gradient).

Objectif du sujet : au moins **98 %** de bonnes réponses sur `dataset_test.csv`.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib seaborn
```

Le venv doit être activé (`source venv/bin/activate`) dans chaque nouveau terminal.

## Utilisation

| script | commande | rôle |
|---|---|---|
| `describe.py` | `python3 describe.py dataset_train.csv` | statistiques de chaque matière (count, mean, std, min, quartiles, max), sans fonction statistique toute faite |
| `histogram.py` | `python3 histogram.py dataset_train.csv` | répartition des notes par maison, pour chaque matière |
| `scatter_plot.py` | `python3 scatter_plot.py dataset_train.csv` | paires de matières les plus corrélées |
| `pair_plot.py` | `python3 pair_plot.py dataset_train.csv` | matrice de nuages de points de toutes les matières |
| `logreg_train.py` | `python3 logreg_train.py dataset_train.csv` | entraîne le modèle et écrit `weights.json` |
| `logreg_predict.py` | `python3 logreg_predict.py dataset_test.csv weights.json` | prédit les maisons et écrit `houses.csv` |

## Visualisation : réponses aux questions du sujet

**Histogramme — quelle matière a une répartition homogène entre les 4 maisons ?**
Care of Magical Creatures (et Arithmancy) : les 4 maisons ont presque la même moyenne et la même dispersion, leurs histogrammes se superposent. Ces matières ne permettent pas de distinguer les maisons.

**Scatter plot — quelles sont les deux features similaires ?**
Astronomy et Defense Against the Dark Arts : corrélation de −1, les points forment une ligne droite (`Defense = Astronomy ÷ −100` pour tous les élèves). Elles donnent la même information.

**Pair plot — quelles features utiliser pour la régression ?**
Celles où les maisons forment des groupes séparés. On retire :
- Arithmancy, Care of Magical Creatures : répartition homogène ;
- Defense Against the Dark Arts : redondante avec Astronomy ;
- Potions : les maisons se chevauchent beaucoup, et la retirer ne change pas la précision.

Features retenues (9) : Astronomy, Herbology, Divination, Muggle Studies, Ancient Runes, History of Magic, Transfiguration, Charms, Flying.

## Fonctionnement de la régression logistique

### Entraînement (`logreg_train.py`)

1. **Nettoyage** : suppression des élèves avec une note manquante dans les features retenues (1600 → 1359 élèves).
2. **Normalisation** : `(note − moyenne) / écart-type` pour chaque matière, avec les fonctions de `describe.py`. Toutes les matières sont ainsi à la même échelle.
3. **Biais** : ajout d'une colonne de 1 → matrice `X` de taille 1359 × 10.
4. **One-vs-all** : pour chaque maison, un classifieur binaire « cette maison ou pas ? » (`y` = 1 pour la maison, 0 sinon).
5. **Descente de gradient** (2000 itérations, α = 0.1), en partant de poids θ à 0 :
   - probabilité : `h = g(Xθ)` avec la sigmoïde `g(z) = 1 / (1 + e^−z)`
   - gradient : `(1/m) · Xᵀ(h − y)`
   - mise à jour : `θ ← θ − α · gradient`
   - le coût `J(θ) = −(1/m) Σ [y·log(h) + (1−y)·log(1−h)]` est affiché pour vérifier qu'il diminue.
6. **Sauvegarde** dans `weights.json` : les features, leurs moyennes et écarts-types, et les 4 vecteurs θ.

### Prédiction (`logreg_predict.py`)

1. Lecture de `weights.json`.
2. Notes manquantes remplacées par la moyenne du train (aucun élève supprimé).
3. Normalisation avec les moyennes et écarts-types **du train**, ajout de la colonne de 1.
4. Probabilité de chaque maison pour chaque élève : `g(X · Θᵀ)`.
5. Maison retenue : celle de plus grande probabilité.
6. Écriture de `houses.csv` :
   ```
   Index,Hogwarts House
   0,Hufflepuff
   1,Ravenclaw
   ...
   ```

## Résultats

- Coût final par maison entre 0.04 et 0.07 (0.69 au départ).
- Précision estimée par validation croisée à 5 plis sur `dataset_train.csv` : **98,2 %**.
