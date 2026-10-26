# Lab experiments

Ten machine learning experiments for ITA 0604, each in its own folder with a notebook, its data (where it needs any) and a README that explains the method and shows the result.

| Experiment | Topic | Tools | Result |
|---|---|---|---|
| [EXP 1](EXP%201/README.md) | Find-S algorithm | Python (`csv`) | Most specific hypothesis `['Sunny', 'Warm', '?', 'Strong', '?', '?']` |
| [EXP 2](EXP%202/README.md) | Candidate Elimination | numpy, pandas | Final S matches EXP 1 |
| [EXP 3](EXP%203/README.md) | ID3 decision tree | numpy, pandas | Tree that splits on Outlook first |
| [EXP 4](EXP%204/README.md) | Decision tree on Iris | scikit-learn | Accuracy 95.6% |
| [EXP 5](EXP%205/README.md) | Perceptron from scratch (AND) | numpy | All four AND cases correct |
| [EXP 6](EXP%206/README.md) | Perceptron with scikit-learn | scikit-learn | Accuracy 88.3% |
| [EXP 7](EXP%207/README.md) | Backpropagation (XOR) | numpy | All four XOR cases correct |
| [EXP 8](EXP%208/README.md) | Multi-layer perceptron on Iris | scikit-learn | Accuracy 95.6% |
| [EXP 9](EXP%209/README.md) | Digit recognition (MNIST) | TensorFlow / Keras | Test accuracy 97.5% |
| [EXP10](EXP10/README.md) | Genetic algorithm | Python (`random`) | Best `x` found: 30 (maximum is 31) |

## Running them

Open any `.ipynb` file in Google Colab or Jupyter. Keep the CSV file in the same folder as the notebook when the experiment uses one (EXP 1, 2 and 3). EXP 9 needs TensorFlow and an internet connection to download the dataset. The other experiments need only `numpy`, `pandas`, `scikit-learn` and `matplotlib`.
