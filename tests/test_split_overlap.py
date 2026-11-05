import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from utils.prepare_data import drop_overlapping_pairs, overlap_key, split_and_save_pairs

LONG_LINE = 'Die Geſchichte des Hauſes'
OTHER_LINE = 'und ſeiner Bewohner im Jahre'


def write_pair(root: Path, folder: str, stem: str, text: str, suffix: str = '.gt.txt') -> tuple[Path, Path, str]:
    directory = root / 'data_sources' / folder
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / f'{stem}.png'
    image.write_bytes(stem.encode())
    text_path = directory / f'{stem}{suffix}'
    text_path.write_text(text, encoding='utf-8')
    return image, text_path, folder


def quietly(function, *args, **kwargs):
    with redirect_stdout(StringIO()):
        return function(*args, **kwargs)


class OverlapKey(unittest.TestCase):
    def test_long_s_hyphen_variants_and_spacing_fold_to_one_key(self):
        self.assertEqual(overlap_key('Die  Geſchichte des Hau⸗'), 'Die Geschichte des Hau-')
        self.assertEqual(overlap_key('Die Geschichte des Hau='), 'Die Geschichte des Hau-')

    def test_case_is_kept(self):
        self.assertNotEqual(overlap_key('die geschichte des hauses'), overlap_key('Die Geschichte des Hauses'))

    def test_lines_shorter_than_the_minimum_are_not_compared(self):
        self.assertIsNone(overlap_key('Amen.'))
        self.assertEqual(overlap_key('Amen, Amen.'), None)
        self.assertEqual(overlap_key('Amen, Amen!!'), 'Amen, Amen!!')


class DropOverlappingPairs(unittest.TestCase):
    def test_every_text_of_an_overlapping_image_is_dropped_and_others_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, gt, folder = write_pair(root, 'cremma', 'p1_l1', OTHER_LINE)
            corrected = image.with_name('p1_l1.corrected.txt')
            corrected.write_text(LONG_LINE, encoding='utf-8')
            kept = write_pair(root, 'cremma', 'p1_l2', OTHER_LINE)
            pairs = [(image, gt, folder), (image, corrected, folder), kept]

            result = quietly(drop_overlapping_pairs, pairs, {overlap_key(LONG_LINE)}, 'training')

        self.assertEqual(result, [kept])

    def test_short_lines_are_kept_even_when_both_sides_share_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            pair = write_pair(Path(tmp), 'cremma', 'p1_l1', 'Amen.')
            self.assertEqual(quietly(drop_overlapping_pairs, [pair], {'Amen.'}, 'training'), [pair])


class SplitAndSave(unittest.TestCase):
    def test_keep_mode_leaves_out_new_training_lines_that_validation_already_holds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_dir = root / 'data' / 'valid'
            valid_dir.mkdir(parents=True)
            (valid_dir / 'catmus_1.gt.txt').write_text(LONG_LINE.replace('ſ', 's'), encoding='utf-8')
            (root / 'data' / 'split_manifest.json').write_text(json.dumps(
                {'train': {}, 'valid': {'catmus_1': 'data_sources/catmus/a.png'}}))
            leaked = write_pair(root, 'cremma', 'page1_l1', LONG_LINE)
            clean = write_pair(root, 'cremma', 'page1_l2', OTHER_LINE)

            quietly(split_and_save_pairs, [leaked, clean], str(root / 'data'), mode='keep', num_workers=1)
            manifest = json.loads((root / 'data' / 'split_manifest.json').read_text())

        self.assertEqual(sorted(manifest['train'].values()), [str(clean[0])])
        self.assertEqual(list(manifest['valid']), ['catmus_1'])

    def test_extend_mode_keeps_new_validation_lines_out_when_training_holds_their_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / 'data' / 'train'
            train_dir.mkdir(parents=True)
            (train_dir / 'catmus_1.gt.txt').write_text(LONG_LINE, encoding='utf-8')
            (train_dir / 'catmus_2.gt.txt').write_text(OTHER_LINE, encoding='utf-8')
            (root / 'data' / 'split_manifest.json').write_text(json.dumps(
                {'train': {'catmus_1': 'a.png', 'catmus_2': 'b.png'}, 'valid': {}}))
            first_page = write_pair(root, 'cremma', 'page1_l1', LONG_LINE)
            second_page = write_pair(root, 'cremma', 'page2_l1', OTHER_LINE)

            quietly(split_and_save_pairs, [first_page, second_page], str(root / 'data'), train_ratio=0.5,
                    mode='extend', num_workers=1)
            manifest = json.loads((root / 'data' / 'split_manifest.json').read_text())

        self.assertEqual(manifest['valid'], {})
        self.assertEqual(len(manifest['train']), 3)


if __name__ == '__main__':
    unittest.main()
