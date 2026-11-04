import unittest

import albumentations as A
import numpy as np
import torch

import train
from utils.augmentations import BaselineCurve, Binarize, NeighbourLineBleed, VerticalCropPad


def ruled_line(height: int = 40, width: int = 60) -> np.ndarray:
    """White paper with one black row at a known height and one grey column."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    image[20] = 0
    image[:, 30] = 128
    return image


class CustomOps(unittest.TestCase):
    def test_padding_the_top_moves_the_ink_down_and_keeps_the_size(self):
        out = VerticalCropPad(p=1.0).apply(ruled_line(), top=0.25, bottom=-0.25)
        self.assertEqual(out.shape, (40, 60, 3))
        self.assertEqual(int(np.argmin(out[:, 0, 0])), 30)
        self.assertTrue(bool((out[:10, 0] == 255).all()))

    def test_no_crop_and_no_pad_is_the_identity(self):
        image = ruled_line()
        self.assertTrue(np.array_equal(VerticalCropPad(p=1.0).apply(image, top=0.0, bottom=0.0), image))

    def test_a_flat_baseline_curve_is_the_identity_and_a_wave_moves_the_ink(self):
        image = ruled_line()
        self.assertTrue(np.array_equal(BaselineCurve(p=1.0).apply(image, amplitude=0.0), image))
        bent = BaselineCurve(p=1.0).apply(image, amplitude=0.1, waves=1.0, phase=np.pi / 2)
        self.assertEqual(bent.shape, image.shape)
        self.assertEqual(int(np.argmin(bent[:, 0, 0])), 16)

    def test_neighbour_bleed_only_darkens_the_edge_it_touches(self):
        image = ruled_line()
        image[36:] = 0
        out = NeighbourLineBleed(p=1.0).apply(image, depth=0.1, from_above=True, offset=0.0)
        self.assertTrue(bool((out[:4] == 0).all()))
        self.assertTrue(np.array_equal(out[4:], image[4:]))

    def test_binarize_leaves_only_black_and_white(self):
        out = Binarize(p=1.0).apply(ruled_line())
        self.assertEqual(out.shape, (40, 60, 3))
        self.assertEqual(sorted(np.unique(out).tolist()), [0, 255])
        self.assertTrue(bool((out[:, 30] == 0).all()))


class TrainingPipeline(unittest.TestCase):
    def setUp(self):
        self.transform, _ = train.get_transforms()

    def test_every_line_keeps_its_height_width_and_bytes(self):
        rng = np.random.default_rng(0)
        self.transform.set_random_seed(0)
        for _ in range(200):
            image = rng.integers(0, 256, size=(train.TARGET_HEIGHT, 384, 3), dtype=np.uint8)
            out = self.transform(image=image)['image']
            self.assertEqual(out.dtype, torch.uint8)
            self.assertEqual(tuple(out.shape), (3, train.TARGET_HEIGHT, 384))

    def test_dropped_operations_are_gone_and_the_new_ones_are_in(self):
        names = {type(t).__name__ for t in self.transform.transforms[0].transforms}
        self.assertTrue({'GridDistortion', 'BaselineCurve', 'VerticalCropPad', 'NeighbourLineBleed',
                         'Downscale', 'ImageCompression', 'Binarize'} <= names)
        self.assertFalse({'CoarseDropout', 'PixelDropout'} & names)
        self.assertIsInstance(self.transform.transforms[0], A.RandomOrder)


if __name__ == '__main__':
    unittest.main()
