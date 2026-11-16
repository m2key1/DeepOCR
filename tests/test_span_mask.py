import unittest

import torch

from utils.model import DeepOCR
from utils.tokenizer import DeepTokenizer

LINES, COLUMNS = 3, 40


def tiny_model() -> DeepOCR:
    torch.manual_seed(0)
    return DeepOCR(tokenizer=DeepTokenizer(normalize_unicode=True, verbose=False), backbone_name='cnn_grouped',
                   cnn_width_stride=8, cnn_patch_size=32, target_height=64, use_2d_structure=False, num_experts=4,
                   top_k_experts=2, expert_ffn='swiglu', expert_hidden=32, router_balance='loss_free',
                   num_encoder_layers=1)


class SpanMask(unittest.TestCase):
    def test_a_zero_ratio_masks_no_column(self):
        model = tiny_model()
        model.set_mask_ratio(0.0)
        mask = model._generate_span_mask(LINES, COLUMNS, torch.device('cpu'))
        self.assertEqual(mask.sum(dim=1).tolist(), [0, 0, 0])

    def test_a_ratio_too_small_for_a_span_still_masks_one(self):
        model = tiny_model()
        # 1% of 40 columns rounds down to none, short of one two-column span
        model.set_mask_ratio(0.01)
        mask = model._generate_span_mask(LINES, COLUMNS, torch.device('cpu'))
        self.assertEqual(model.span_length, 2)
        self.assertEqual(mask.sum(dim=1).tolist(), [2, 2, 2])


if __name__ == '__main__':
    unittest.main()
