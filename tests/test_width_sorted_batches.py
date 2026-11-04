import unittest

import numpy as np
import torch
from PIL import Image

from utils.dataset import WidthSortedValidationSampler
from utils.model import DeepOCR
from utils.tokenizer import DeepTokenizer

WIDTHS = [50, 10, 40, 30, 20, 60, 70]


class ValidationBatches(unittest.TestCase):
    def test_one_gpu_reads_every_line_once_in_width_order(self):
        batches = list(WidthSortedValidationSampler(WIDTHS, batch_size=3))
        self.assertEqual(batches, [[1, 4, 3], [2, 0, 5], [6]])

    def test_gpus_share_the_batches_without_dropping_or_repeating_a_line(self):
        shards = [list(WidthSortedValidationSampler(WIDTHS, batch_size=2, num_replicas=3, rank=rank))
                  for rank in range(3)]
        self.assertEqual(shards, [[[1, 4], [6]], [[3, 2]], [[0, 5]]])
        self.assertEqual([len(WidthSortedValidationSampler(WIDTHS, batch_size=2, num_replicas=3, rank=rank))
                          for rank in range(3)], [2, 1, 1])

    def test_ties_keep_the_dataset_order(self):
        batches = list(WidthSortedValidationSampler([5, 5, 1, 5], batch_size=4))
        self.assertEqual(batches, [[2, 0, 1, 3]])


class BatchPrediction(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = DeepOCR(tokenizer=DeepTokenizer(normalize_unicode=True, verbose=False),
                             backbone_name='cnn_grouped', cnn_width_stride=8, cnn_patch_size=32, target_height=64,
                             use_2d_structure=False, num_experts=4, top_k_experts=2, expert_ffn='swiglu',
                             expert_hidden=32, router_balance='loss_free', num_encoder_layers=1).eval()
        rng = np.random.default_rng(0)
        self.images = [Image.fromarray(rng.integers(0, 256, size=(32, width, 3), dtype=np.uint8))
                       for width in (90, 20, 61)]
        self.seen = []
        encode = self.model._encode

        def recording_encode(x, attention_masks=None, **kwargs):
            self.seen.append((x.clone(), attention_masks.clone()))
            return encode(x, attention_masks=attention_masks, **kwargs)

        self.model._encode = recording_encode

    def test_lines_are_batched_narrowest_first_as_bytes_padded_white(self):
        self.model.predict(self.images, device=torch.device('cpu'), batch_size=3)
        batch, masks = self.seen[0]
        self.assertEqual(batch.dtype, torch.uint8)
        self.assertEqual(tuple(batch.shape), (3, 3, 64, 184))
        self.assertTrue(bool((batch[0, :, :, 40:] == 255).all()))
        self.assertEqual((~masks).sum(dim=1).tolist(), [5, 16, 23])

    def test_answers_come_back_in_the_order_given(self):
        forward = self.model.predict(self.images, device=torch.device('cpu'), batch_size=3)
        backward = self.model.predict(self.images[::-1], device=torch.device('cpu'), batch_size=3)
        self.assertEqual(forward, backward[::-1])
        self.assertEqual(len(forward), 3)


if __name__ == '__main__':
    unittest.main()
