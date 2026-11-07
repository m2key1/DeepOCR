import tempfile
import unittest
from pathlib import Path

from PIL import Image

from utils.dataset import OCRDataset
from utils.tokenizer import DeepTokenizer


class LineImageFormats(unittest.TestCase):
    def test_tif_lines_are_read_beside_png_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            Image.new('RGB', (400, 32), 'white').save(folder / 'a.png')
            (folder / 'a.gt.txt').write_text('Haus', encoding='utf-8')
            Image.new('1', (400, 32), 1).save(folder / 'b.tif')
            (folder / 'b.gt.txt').write_text('Hof', encoding='utf-8')
            dataset = OCRDataset(folder, DeepTokenizer(verbose=False), target_height=32, patch_size=32,
                                 patch_size_w=8, use_2d_structure=False)
            self.assertEqual(sorted(text for _, text, _ in dataset.data_pairs), ['Haus', 'Hof'])
            self.assertEqual(dataset[1][0].shape, dataset[0][0].shape)


if __name__ == '__main__':
    unittest.main()
