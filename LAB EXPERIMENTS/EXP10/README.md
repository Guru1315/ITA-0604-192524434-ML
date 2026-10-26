# EXP 10: Genetic algorithm

A simple **genetic algorithm** that searches for the integer `x` between 0 and 31 that maximises `f(x) = x²`.

## Files

| File | What it is |
|---|---|
| `EXP_10.ipynb` | The genetic algorithm and one saved run |

## How it works

- **Chromosome:** a 5-bit binary string, which `decode` turns into an integer from 0 to 31.
- **Fitness:** the square of the decoded number, so bigger numbers are fitter.
- **Population:** 6 chromosomes, evolved for 20 generations.
- **Selection:** roulette-wheel, where fitter chromosomes are more likely to be chosen as parents.
- **Crossover:** single-point, at a random position, producing two children from two parents.
- **Mutation:** each bit flips with probability 0.05.

## Result

```
Best x = 30, f(x) = 900
```

The random generator is not seeded, so the answer changes from run to run. The true maximum is `x = 31`, where `f(x) = 961`, so a run can land close to it without reaching it.

## Run it

Open `EXP_10.ipynb` in Jupyter or Google Colab and run the cell. It uses only Python's standard library.
