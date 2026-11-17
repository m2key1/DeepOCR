import unittest
from types import SimpleNamespace

import train
from utils.model import DeepOCR
from utils.tokenizer import DeepTokenizer


class Strides(unittest.TestCase):
    def test_a_model_reading_other_patches_than_the_datasets_assume_is_refused(self):
        transformer = SimpleNamespace(patch_size=16, patch_size_w=16)
        with self.assertRaisesRegex(ValueError, '32 x 32 pixel patches, but the model reads 16 x 16'):
            train.check_strides(transformer, 32, 32)

    def test_the_default_backbone_reads_the_patches_the_datasets_assume(self):
        model = DeepOCR(tokenizer=DeepTokenizer(normalize_unicode=True, verbose=False), backbone_name=train.BACKBONE,
                        cnn_width_stride=train.CNN_WIDTH_STRIDE, cnn_patch_size=train.PATCH_SIZE,
                        target_height=train.TARGET_HEIGHT, use_2d_structure=False, num_experts=4, top_k_experts=2,
                        expert_ffn='swiglu', expert_hidden=32, router_balance='loss_free', num_encoder_layers=1)
        train.check_strides(model, train.PATCH_SIZE, train.WIDTH_STRIDE)
        self.assertEqual((train.PATCH_SIZE, train.WIDTH_STRIDE), (32, 8))


if __name__ == '__main__':
    unittest.main()
