# EMNIST Conv-MLP Classifier

A finished PyTorch project for classifying handwritten characters from the
[EMNIST ByClass](https://www.nist.gov/itl/products-and-services/emnist-dataset)
dataset. The model combines a compact convolutional feature extractor with a
multilayer perceptron (MLP) classifier.

## Highlights

- 62-class character classification (`0-9`, `A-Z`, and `a-z`)
- Convolutional layers learn spatial stroke features before the MLP makes the
  final prediction
- Reproducible train/evaluate commands with automatic CPU or CUDA selection
- Checkpoints include the model configuration

## Project layout

```text
emnist-conv-mlp-classifier/
├── src/emnist_classifier/   # model, data utilities, and training code
├── tests/                  # lightweight architecture checks
├── README.md
├── pyproject.toml
└── .gitignore
```

## Quick start

Create an environment with Python 3.10+ and install the project. Install the
appropriate PyTorch build for your hardware first, following the
[official PyTorch instructions](https://pytorch.org/get-started/locally/).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Train the default model (the dataset is downloaded to `data/` on first run):

```bash
python3 -m emnist_classifier.train --epochs 10 --batch-size 128
```

Evaluate a saved checkpoint:

```bash
python3 -m emnist_classifier.evaluate --checkpoint artifacts/best_model.pt
```

Run the architecture tests:

```bash
python3 -m unittest discover -s tests -v
```

## Model

Input images are grayscale `28 × 28` EMNIST characters. Two convolutional
blocks progressively learn local stroke patterns and reduce the spatial size.
The resulting feature map is flattened and passed through two fully connected
layers with dropout before producing 62 output logits.

```text
1×28×28 → Conv(32) → Conv(64) → AdaptiveAvgPool → Flatten
        → Linear(64→128) → Dropout → Linear(128→62)
```

The training objective is cross-entropy loss, optimized with AdamW. Accuracy
is measured from the highest-logit predicted class.

## Notes

EMNIST labels follow the `byclass` split's ordering. Training artefacts and
downloaded data are intentionally excluded from version control so the
repository remains small and easy to clone.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

