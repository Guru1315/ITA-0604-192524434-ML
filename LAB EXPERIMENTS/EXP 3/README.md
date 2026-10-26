# EXP 3: ID3 decision tree

Builds a decision tree from scratch with the **ID3 algorithm**, which splits on the attribute with the highest information gain.

## Files

| File | What it is |
|---|---|
| `EXP_3.ipynb` | The ID3 code and the tree it prints |
| `playtennis.csv` | The training data used by the notebook (14 days) |
| `enjoysport.csv` | A copy of the EXP 1 and EXP 2 data, not used by this notebook |

## Data

`playtennis.csv` has four attributes (`Outlook, Temperature, Humidity, Wind`) and the label `PlayTennis` (Yes or No).

## How it works

- `entropy(col)` measures how mixed the labels are.
- `info_gain(data, attr, target)` is the drop in entropy after splitting on an attribute.
- `id3(data, attrs, target)` picks the best attribute, makes a branch for each of its values and repeats on each branch. It stops when all labels in a branch agree, or when no attributes are left.

## Result

```
Decision Tree: {'Outlook': {'Overcast': 'Yes',
                            'Rain':  {'Wind': {'Strong': 'No', 'Weak': 'Yes'}},
                            'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}}}}
```

Read it as: if the outlook is overcast, play. If it is rainy, play only when the wind is weak. If it is sunny, play only when the humidity is normal.

## Run it

Open `EXP_3.ipynb` in Jupyter or Google Colab, keep `playtennis.csv` in the same folder, and run the cell. It needs `numpy` and `pandas`.
