import unittest
import torch
from amgm.models.neural_SDE.gate_regularizers import (
    entropy_regularization,
    adaptive_entropy_regularization,
    regional_variance_regularization,
    mutual_information_regularization
)

class TestGateRegularizers(unittest.TestCase):
    def setUp(self):
        # Batch size 10, 2 experts. All samples go to Expert 0.
        self.collapsed_pi = torch.zeros((10, 2))
        self.collapsed_pi[:, 0] = 1.0

        # Batch size 10, 2 experts. Perfectly balanced (0.5 for all).
        self.balanced_pi = torch.ones((10, 2)) * 0.5

        # Batch size 10, 2 experts. First 5 go to Expert 0, last 5 go to Expert 1.
        self.specialized_pi = torch.zeros((10, 2))
        self.specialized_pi[:5, 0] = 1.0
        self.specialized_pi[5:, 1] = 1.0

    def test_entropy_regularization(self):
        loss_c = entropy_regularization(self.collapsed_pi)
        self.assertTrue(torch.isclose(loss_c, torch.tensor(0.0), atol=1e-5))

        loss_b = entropy_regularization(self.balanced_pi)
        self.assertTrue(torch.isclose(loss_b, torch.tensor(0.693147), atol=1e-4))

    def test_adaptive_entropy_regularization(self):
        loss_c = adaptive_entropy_regularization(self.collapsed_pi, clip=0.1)
        self.assertTrue(torch.isclose(loss_c, torch.tensor(0.1), atol=1e-5))

        loss_b = adaptive_entropy_regularization(self.balanced_pi, clip=0.1)
        self.assertTrue(torch.isclose(loss_b, torch.tensor(0.0), atol=1e-5))

        loss_s = adaptive_entropy_regularization(self.specialized_pi, clip=0.1)
        self.assertTrue(torch.isclose(loss_s, torch.tensor(0.0), atol=1e-5))

    def test_regional_variance_regularization(self):
        loss_c = regional_variance_regularization(self.collapsed_pi, num_regions=2)
        self.assertTrue(torch.isclose(loss_c, torch.tensor(0.0), atol=1e-5))

        loss_b = regional_variance_regularization(self.balanced_pi, num_regions=2)
        self.assertTrue(torch.isclose(loss_b, torch.tensor(0.693147), atol=1e-4))

    def test_mutual_information_regularization(self):
        loss_c = mutual_information_regularization(self.collapsed_pi)
        self.assertTrue(torch.isclose(loss_c, torch.tensor(0.0), atol=1e-5))
        
        loss_b = mutual_information_regularization(self.balanced_pi)
        self.assertTrue(torch.isclose(loss_b, torch.tensor(0.0), atol=1e-5))
        
        loss_s = mutual_information_regularization(self.specialized_pi)
        self.assertTrue(torch.isclose(loss_s, torch.tensor(0.693147), atol=1e-4))

if __name__ == '__main__':
    unittest.main()
