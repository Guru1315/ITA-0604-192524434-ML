# EXP 7: Backpropagation from scratch (XOR)

A small neural network written with NumPy that learns the **XOR** function, which a single perceptron cannot learn.

## Files

| File | What it is |
|---|---|
| `EXP_7.ipynb` | The network, its training loop and the final predictions |

## How it works

- **Architecture:** 2 inputs, a hidden layer of 4 neurons, 1 output, with sigmoid activations.
- **Training:** 10,000 epochs. Each epoch does a forward pass, works out the error at the output and the hidden layer, and updates both weight matrices with a learning rate of 0.5.
- The random weights are seeded (`np.random.seed(1)`), so the run is repeatable.

## Result

```
Predicted output after training:
 [[0.032]
  [0.975]
  [0.971]
  [0.028]]
```

For the inputs `(0,0), (0,1), (1,0), (1,1)` the targets are `0, 1, 1, 0`, and the outputs round to exactly those values.

## Run it

Open `EXP_7.ipynb` in Jupyter or Google Colab and run the cell. It needs only `numpy`.
