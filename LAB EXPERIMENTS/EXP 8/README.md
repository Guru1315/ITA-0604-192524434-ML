# EXP 8: Multi-layer perceptron on the Iris dataset

Trains a small neural network with scikit-learn's `MLPClassifier`.

## Files

| File | What it is |
|---|---|
| `EXP_8.ipynb` | The code and the accuracy it prints |

## How it works

1. Load the Iris dataset and split it 70% training, 30% testing (`random_state=1`).
2. Scale the features with `StandardScaler`, fitted on the training data only.
3. Train an `MLPClassifier` with one hidden layer of 10 neurons (`max_iter=1000`).
4. Print the accuracy on the test set.

## Result

```
Accuracy: 0.9555555555555556
```

That is 43 of the 45 test flowers classified correctly, the same accuracy as the decision tree in [EXP 4](../EXP%204/README.md).

## Run it

Open `EXP_8.ipynb` in Jupyter or Google Colab and run the cell. It needs `scikit-learn`.
