import unittest

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

import train
from utils.dataset import make_collate_fn
from utils.model import DeepOCR, normalize_images
from utils.tokenizer import DeepTokenizer


def line_image(seed: int, width: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(64, width, 3), dtype=np.uint8)


class UnsignedBytesReachTheModel(unittest.TestCase):
    def test_transforms_hand_over_raw_bytes(self):
        train_transform, val_transform = train.get_transforms()
        image = line_image(0, 96)
        for transform in (train_transform, val_transform):
            tensor = transform(image=image)['image']
            self.assertEqual(tensor.dtype, torch.uint8)
            self.assertEqual(tuple(tensor.shape), (3, 64, 96))
        self.assertTrue(torch.equal(val_transform(image=image)['image'], torch.from_numpy(image).permute(2, 0, 1)))

    def test_normalizing_on_the_gpu_side_matches_the_old_cpu_normalization(self):
        image = line_image(1, 80)
        old = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
        expected = old(image=image)['image']
        actual = normalize_images(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0))[0]
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_normalized_input_passes_through_unchanged(self):
        normalized = torch.randn(2, 3, 64, 80)
        self.assertIs(normalize_images(normalized), normalized)

    def test_the_encoder_reads_bytes_as_it_reads_normalized_floats(self):
        torch.manual_seed(0)
        model = DeepOCR(tokenizer=DeepTokenizer(normalize_unicode=True, verbose=False), backbone_name='cnn_grouped',
                        cnn_width_stride=8, cnn_patch_size=32, target_height=64, use_2d_structure=False,
                        num_experts=4, top_k_experts=2, expert_ffn='swiglu', expert_hidden=32,
                        router_balance='loss_free', num_encoder_layers=1).eval()
        raw = torch.stack([torch.from_numpy(line_image(seed, 128)).permute(2, 0, 1) for seed in (2, 3)])
        with torch.no_grad():
            from_bytes = model._encode(raw)[0]
            from_floats = model._encode(normalize_images(raw))[0]
        self.assertTrue(torch.equal(from_bytes, from_floats))

    def test_collate_keeps_bytes_and_pads_with_white(self):
        collate = make_collate_fn(patch_size=64, patch_size_w=8, use_2d_structure=False, fill_value=train.PAD_WHITE,
                                  collapse_height=True)
        narrow = torch.from_numpy(line_image(4, 40)).permute(2, 0, 1)
        wide = torch.from_numpy(line_image(5, 64)).permute(2, 0, 1)
        images = collate([(narrow, torch.tensor([5]), 'a', None), (wide, torch.tensor([6]), 'b', None)])[0]
        self.assertEqual(images.dtype, torch.uint8)
        self.assertTrue(torch.equal(images[0, :, :, :40], narrow))
        self.assertTrue(bool((images[0, :, :, 40:] == 255).all()))
        self.assertTrue(torch.equal(images[1], wide))


if __name__ == '__main__':
    unittest.main()
