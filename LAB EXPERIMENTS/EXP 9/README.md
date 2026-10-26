# EXP 9: Handwritten digit recognition with a neural network (MNIST)

Trains a fully connected neural network in TensorFlow/Keras to recognise handwritten digits.

## Files

| File | What it is |
|---|---|
| `EXP_9.ipynb` | The code, the training log and the test accuracy |

## How it works

1. Download the MNIST dataset (60,000 training and 10,000 test images of digits, 28 by 28 pixels) and scale the pixel values to 0 to 1.
2. Build a `Sequential` model: flatten the image, a Dense layer of 128 neurons (ReLU), a Dense layer of 64 neurons (ReLU) and a 10-way softmax output.
3. Compile with the `adam` optimiser and `sparse_categorical_crossentropy` loss.
4. Train for 5 epochs, keeping 10% of the training images for validation.
5. Evaluate on the 10,000 test images.

## Result

| Epoch | Training accuracy | Validation accuracy |
|---|---|---|
| 1 | 0.9237 | 0.9662 |
| 5 | 0.9854 | 0.9777 |

```
Test accuracy: 0.9746
```

## Run it

Open `EXP_9.ipynb` in Google Colab (easiest, because TensorFlow is already installed) or in Jupyter with `tensorflow` installed. The dataset is downloaded automatically, so an internet connection is needed. A warning about `input_shape` appears in the output; it is harmless.
