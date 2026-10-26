# EXP 2: Candidate Elimination algorithm

Keeps track of two boundaries of hypotheses, the **specific boundary S** and the **general boundary G**, while reading the training examples.

## Files

| File | What it is |
|---|---|
| `EXP_2.ipynb` | A simplified Candidate Elimination implementation and its output |
| `enjoysport.csv` | The training data (4 examples, same as EXP 1) |

## How it works

- `S` starts as the first example and `G` starts as the fully general hypothesis (`?` in every position).
- For a **positive** example, hypotheses in `G` that do not cover it are dropped, and `S` is generalised just enough to cover it.
- For a **negative** example, hypotheses in `G` are made more specific, using values taken from `S`.
- The helper `more_general(h1, h2)` checks whether one hypothesis covers another.

## Result printed by the notebook

```
Final S: [['Sunny', 'Warm', '?', 'Strong', '?', '?']]
Final G: [['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]
```

The final `S` is the same hypothesis that Find-S gave in [EXP 1](../EXP%201/README.md).

## Run it

Open `EXP_2.ipynb` in Jupyter or Google Colab, keep `enjoysport.csv` in the same folder, and run the cell. It needs `numpy` and `pandas`.
