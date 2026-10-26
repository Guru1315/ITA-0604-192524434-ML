# EXP 5: Perceptron from scratch (AND gate)

Writes the perceptron learning rule by hand with NumPy and teaches it the logical **AND** function.

## Files

| File | What it is |
|---|---|
| `EXP_5.ipynb` | The training function, learned weights and predictions |

## How it works

- `perceptron_train(X, y, epochs=20, lr=0.1)` starts with zero weights (the first weight is the bias).
- For every example it predicts 1 if the weighted sum is above 0, otherwise 0, then nudges the weights by `lr * (target - prediction) * input`.
- It is trained on the four AND inputs: `(0,0), (0,1), (1,0), (1,1)` with targets `0, 0, 0, 1`.

## Result

```
Learned weights (AND): [-0.2  0.2  0.1]    (bias, w1, w2)
[0 0] -> 0
[0 1] -> 0
[1 0] -> 0
[1 1] -> 1
```

All four AND cases are predicted correctly.

## Run it

Open `EXP_5.ipynb` in Jupyter or Google Colab and run the cell. It needs only `numpy`.
