# EXP 4: Decision tree classifier on the Iris dataset

Trains a decision tree with scikit-learn and draws the tree.

## Files

| File | What it is |
|---|---|
| `EXP_4.ipynb` | The code, its accuracy and the tree diagram (stored inside the notebook output) |

## How it works

1. Load the built-in Iris dataset (150 flowers, 4 measurements, 3 species).
2. Split it 70% for training and 30% for testing (`random_state=1`).
3. Train a `DecisionTreeClassifier` that splits using **entropy**.
4. Print the accuracy on the test set and draw the tree with `plot_tree`, which is also saved as `iris_tree.png` when the notebook runs.

## Result

```
Accuracy: 0.9555555555555556
```

That is 43 of the 45 test flowers classified correctly.

## Run it

Open `EXP_4.ipynb` in Jupyter or Google Colab and run the cell. It needs `scikit-learn` and `matplotlib`. The image `iris_tree.png` is created by running the notebook and is not stored separately in this repository.
