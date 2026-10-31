import unittest
from pathlib import Path

from utils.dataset import derive_dataset_label
from utils.prepare_data import _pair_matches_sources

SOURCES = ['data_sources/Schematismus/train', 'data_sources/party_zenodo/10255840']
PAGE_CROP = 'data_sources/_page_lines/10255840/train/page/fmgh021_1_tr_1_tl_22.png'


class DatasetLabels(unittest.TestCase):
    def test_a_page_crop_takes_the_label_of_its_page_lines_folder(self):
        self.assertEqual(derive_dataset_label(PAGE_CROP, SOURCES), '10255840')

    def test_a_train_folder_inside_another_corpus_does_not_match_an_ambiguous_leaf(self):
        self.assertEqual(derive_dataset_label('data_sources/_page_lines/Boccace/train/a.png',
                                              ['data_sources/Schematismus/train']), '_page_lines_Boccace')

    def test_the_source_itself_still_gets_its_qualified_label(self):
        self.assertEqual(derive_dataset_label('data_sources/Schematismus/train/1.png', SOURCES),
                         'Schematismus_train')


class PairMatchesSources(unittest.TestCase):
    def test_a_page_crop_matches_only_the_source_it_came_from(self):
        pair = (Path(PAGE_CROP), Path('x.gt.txt'), 'text')
        self.assertTrue(_pair_matches_sources(pair, ['data_sources/party_zenodo/10255840']))
        self.assertFalse(_pair_matches_sources(pair, ['data_sources/Schematismus/train']))


if __name__ == '__main__':
    unittest.main()
