import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from utils.dataset import OCRDataset
from utils.tokenizer import DeepTokenizer


class NegativeLines(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        folder = Path(self.folder.name)
        pixels = np.full((32, 96, 3), 200, dtype=np.uint8)
        pixels[8:24, 10:30] = 0
        pixels[0:4, :] = 180
        Image.fromarray(pixels).save(folder / 'line.png')
        (folder / 'line.gt.txt').write_text('Haus', encoding='utf-8')
        self.dataset = OCRDataset(folder, DeepTokenizer(verbose=False), target_height=32, patch_size=32,
                                  patch_size_w=8, use_2d_structure=False, negative_prob=0.5)
        self.pixels = pixels

    def tearDown(self):
        self.folder.cleanup()

    def item(self, draws):
        with mock.patch('utils.dataset.random.random', side_effect=draws):
            image, target, _, _ = self.dataset[0]
        # the dataset hands images over channels first
        return np.asarray(image).transpose(1, 2, 0), target.tolist()

    def test_a_line_drawn_above_the_share_reads_as_written(self):
        image, target = self.item([0.7])
        self.assertEqual(target, DeepTokenizer(verbose=False).encode('Haus'))
        np.testing.assert_array_equal(image, self.pixels)

    def test_an_upside_down_line_reads_as_nothing(self):
        image, target = self.item([0.2, 0.3])
        self.assertEqual(target, [])
        np.testing.assert_array_equal(image, self.pixels[::-1, ::-1])

    def test_bare_paper_in_the_lines_colour_reads_as_nothing(self):
        image, target = self.item([0.2, 0.8])
        self.assertEqual(target, [])
        np.testing.assert_array_equal(image, np.full((32, 96, 3), 200, dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
