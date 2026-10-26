# EXP 1: Find-S algorithm

Finds the **most specific hypothesis** that fits the positive examples in a small training set.

## Files

| File | What it is |
|---|---|
| `EXP_1.ipynb` | The Find-S code and its output |
| `enjoysport.csv` | The training data (4 examples) |

## Data

Each row describes a day with six attributes (`Sky, AirTemp, Humidity, Wind, Water, Forecast`) and whether the person enjoyed the sport (`EnjoySport`: Yes or No). Three of the four examples are positive.

## How it works

1. Start with the most specific hypothesis, `['0', '0', '0', '0', '0', '0']`.
2. Go through the rows and ignore every negative (`No`) example.
3. For each positive example, copy a value into the hypothesis if that position is still `0`. If the position already holds a different value, replace it with `?` (any value is fine).

## Result

```
Most specific hypothesis: ['Sunny', 'Warm', '?', 'Strong', '?', '?']
```

In words: the sport is enjoyed when the sky is sunny, the air is warm and the wind is strong, whatever the humidity, water and forecast.

## Run it

Open `EXP_1.ipynb` in Jupyter or Google Colab, keep `enjoysport.csv` in the same folder, and run the cell. Only the Python standard library (`csv`) is needed.
