"""Fast checks that do not need the EMNIST download."""

import unittest

import torch

from emnist_classifier import ConvMLPClassifier, ModelConfig


class ConvMLPClassifierTests(unittest.TestCase):
    def test_output_shape_for_emnist_batch(self) -> None:
        model = ConvMLPClassifier()
        logits = model(torch.randn(8, 1, 28, 28))
        self.assertEqual(logits.shape, (8, 62))

    def test_custom_number_of_classes(self) -> None:
        model = ConvMLPClassifier(ModelConfig(num_classes=10))
        logits = model(torch.randn(2, 1, 28, 28))
        self.assertEqual(logits.shape, (2, 10))


if __name__ == "__main__":
    unittest.main()

