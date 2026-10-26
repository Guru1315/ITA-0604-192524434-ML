# EXP 6: Perceptron classifier with scikit-learn

Trains scikit-learn's ready-made `Perceptron` on a made-up two-class dataset.

## Files

| File | What it is |
|---|---|
| `EXP_6.ipynb` | The code and the accuracy it prints |

## How it works

1. Generate 200 points with 2 features using `make_classification` (`random_state=1`).
2. Split 70% for training and 30% for testing.
3. Train `Perceptron(max_iter=1000, eta0=0.1)` and measure accuracy on the test points.

## Result

```
Accuracy: 0.8833333333333333
```

That is 53 of the 60 test points classified correctly.

## Run it

Open `EXP_6.ipynb` in Jupyter or Google Colab and run the cell. It needs `scikit-learn`.
