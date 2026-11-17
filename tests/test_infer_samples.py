import tempfile
import unittest
from pathlib import Path
from unittest import mock

import infer


class EvaluationSample(unittest.TestCase):
    def collect(self, folder: Path) -> list[str]:
        with mock.patch.multiple(infer, VALID_DIR=folder, MANIFEST_PATH=folder / 'missing.json', MAX_SAMPLES=4):
            return [image_path.stem for image_path, _, _ in infer.collect_samples()]

    def test_the_sample_is_drawn_from_every_corpus_and_is_the_same_each_time(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            for corpus in ('alpha', 'beta'):
                for index in range(6):
                    (folder / f'{corpus}_{index}.png').touch()
                    (folder / f'{corpus}_{index}.gt.txt').write_text(f'{corpus} {index}', encoding='utf-8')
            (folder / 'alpha_9.png').touch()
            first, second = self.collect(folder), self.collect(folder)
        self.assertEqual(first, ['beta_0', 'beta_5', 'alpha_0', 'alpha_4'])
        self.assertEqual(second, first)


if __name__ == '__main__':
    unittest.main()
