import os
import random
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from utils.conventions import (ConventionTable, hold_trained_flags_to_rate, normalize_long_s,
                               normalize_typographic_punctuation,
                               remap_convention_columns, restore_long_s_by_position,
                               restore_long_s_german)
from utils.dataset import Collate, OCRDataset
from utils.tokenizer import DeepTokenizer

NAMES = ['long_s', 'oblique_hyphen', 'app_defined']


def table(image_states=None, flip_prob=1.0):
    return ConventionTable(NAMES, {'diplomatic': [1.0, 1.0, 1.0]}, image_states, flip_prob)


class RestoreLongS(unittest.TestCase):
    def test_position_rule_keeps_round_s_only_at_word_end(self):
        self.assertEqual(restore_long_s_by_position('Das Wasser des Geistes'), 'Das Waſſer des Geiſtes')

    def test_position_rule_treats_a_line_end_hyphen_as_inside_the_word(self):
        self.assertEqual(restore_long_s_by_position('und des-'), 'und deſ-')
        self.assertEqual(restore_long_s_by_position('Haus-Tür'), 'Haus-Tür')

    def test_position_rule_writes_round_s_before_an_apostrophe(self):
        self.assertEqual(restore_long_s_by_position("Gott's"), "Gott's")

    def test_german_rule_writes_round_s_before_consonants_outside_its_set(self):
        self.assertEqual(restore_long_s_german('Glasbläser Dienstag'), 'Glasbläſer Dienſtag')
        self.assertEqual(restore_long_s_by_position('Glasbläser Dienstag'), 'Glaſbläſer Dienſtag')

    def test_capital_s_is_never_touched(self):
        self.assertEqual(restore_long_s_by_position('Sonne'), 'Sonne')


class Normalize(unittest.TestCase):
    def test_long_s_normalizes_without_touching_other_marks(self):
        self.assertEqual(normalize_long_s('Waſſer⸗'), 'Wasser⸗')

    def test_typographic_punctuation_maps_to_ascii(self):
        self.assertEqual(normalize_typographic_punctuation('‘a’ ”b” 1–2'), "'a' \"b\" 1-2")


class Sample(unittest.TestCase):
    def test_a_diplomatic_line_is_normalized_and_its_flags_cleared(self):
        self.assertEqual(table().sample('diplomatic', 'Waſſer⸗'), ('Wasser-', [0.0, 0.0, 1.0]))

    def test_a_corpus_without_the_feature_in_print_flips_with_the_text_unchanged(self):
        result = table({'long_s': {'antiqua': 'absent'}}).sample('antiqua', 'Das Wasser')
        self.assertEqual(result, ('Das Wasser', [1.0, 0.0, 0.0]))

    def test_a_corpus_printing_the_feature_flips_with_its_restorer(self):
        result = table({'long_s': {'fraktur': 'german'}}).sample('fraktur', 'Glasbläser')
        self.assertEqual(result, ('Glasbläſer', [1.0, 0.0, 0.0]))

    def test_printed_without_a_trusted_rule_and_unknown_corpora_stay_as_they_are(self):
        states = table({'long_s': {'manuscript': 'printed'}})
        self.assertEqual(states.sample('manuscript', 'Das Wasser'), ('Das Wasser', [0.0, 0.0, 0.0]))
        self.assertEqual(states.sample('unknown', 'Das Wasser'), ('Das Wasser', [0.0, 0.0, 0.0]))

    def test_no_flip_probability_returns_the_corpus_flags(self):
        self.assertEqual(table(flip_prob=0.0).sample('diplomatic', 'Waſſer⸗'), ('Waſſer⸗', [1.0, 1.0, 1.0]))

    def test_each_flag_flips_independently(self):
        rng = random.Random(0)
        outcomes = {table(flip_prob=0.5).sample('diplomatic', 'Waſſer⸗', rng=rng)[0] for _ in range(200)}
        self.assertEqual(outcomes, {'Waſſer⸗', 'Wasser⸗', 'Waſſer-', 'Wasser-'})

    def test_a_line_with_an_alternate_target_flips_to_it(self):
        states = ConventionTable(['long_s', 'expand_abbreviations'], {}, flip_prob=1.0)
        result = states.sample('compass', 'Gschf.: Haus', {'expand_abbreviations': 'Geschäftsführer: Haus'})
        self.assertEqual(result, ('Geschäftsführer: Haus', [0.0, 1.0]))

    def test_other_flags_still_transform_the_alternate_target(self):
        states = ConventionTable(['long_s', 'expand_abbreviations'], {'fraktur': [1.0, 0.0]}, flip_prob=1.0)
        result = states.sample('fraktur', 'Geſ. m. b. H.', {'expand_abbreviations': 'Geſellſchaft'})
        self.assertEqual(result, ('Gesellschaft', [0.0, 1.0]))

    def test_without_flipping_the_alternate_target_is_ignored(self):
        states = ConventionTable(['expand_abbreviations'], {}, flip_prob=0.0)
        self.assertEqual(states.sample('compass', 'Gschf.', {'expand_abbreviations': 'Geschäftsführer'}),
                         ('Gschf.', [0.0]))

    def test_a_line_read_from_its_image_trains_under_every_flag_combination(self):
        states = ConventionTable(['long_s', 'oblique_hyphen'], {'kurrent': [0.0, 0.0]}, flip_prob=0.5)
        rng = random.Random(0)
        outcomes = {(text, tuple(flags)) for text, flags in
                    (states.sample('kurrent', 'Wasser-', {'diplomatic': 'Waſſer⸗'}, rng=rng) for _ in range(200))}
        self.assertEqual(outcomes, {('Waſſer⸗', (1.0, 1.0)), ('Wasser⸗', (0.0, 1.0)),
                                    ('Waſſer-', (1.0, 0.0)), ('Wasser-', (0.0, 0.0))})

    def test_a_flag_the_corpus_already_keeps_stays_as_transcribed(self):
        states = ConventionTable(['long_s', 'typographic_punctuation'], {'hsa': [0.0, 1.0]}, flip_prob=0.5)
        rng = random.Random(1)
        texts = {states.sample('hsa', '„Haus“ – Wasser', {'diplomatic': '„Haus“ – Waſſer'}, rng=rng)[0]
                 for _ in range(200)}
        self.assertEqual(texts, {'„Haus“ – Waſſer', '„Haus“ – Wasser', '„Haus“ - Waſſer', '„Haus“ - Wasser'})

    def test_without_flipping_the_line_read_from_its_image_is_ignored(self):
        states = ConventionTable(['long_s'], {'kurrent': [0.0]}, flip_prob=0.0)
        self.assertEqual(states.sample('kurrent', 'Wasser', {'diplomatic': 'Waſſer'}), ('Wasser', [0.0]))

    def test_an_alternate_target_replaces_the_marks_read_from_the_image(self):
        states = ConventionTable(['long_s', 'expand_abbreviations'], {'fraktur': [0.0, 0.0]}, flip_prob=1.0)
        result = states.sample('fraktur', 'Ges. m. b. H.',
                               {'diplomatic': 'Geſ. m. b. H.', 'expand_abbreviations': 'Gesellschaft'})
        self.assertEqual(result, ('Gesellschaft', [0.0, 1.0]))

    def test_an_unknown_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "state 'yes' for 'fraktur'"):
            table({'long_s': {'fraktur': 'yes'}})

    def test_image_state_labels_without_samples_are_reported(self):
        states = table({'long_s': {'fraktur': 'german', 'frakture': 'german'}})
        self.assertEqual(states.unseen_labels({'fraktur', 'diplomatic'}), {'long_s': ['frakture']})


class AlternateTargetsOnDisk(unittest.TestCase):
    def test_the_dataset_reads_each_known_flags_alternate_file(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            Image.new('RGB', (400, 32), 'white').save(folder / 'line.png')
            (folder / 'line.gt.txt').write_text('Gschf.: Haus', encoding='utf-8')
            (folder / 'line.expand_abbreviations.txt').write_text('Geschäftsführer: Haus', encoding='utf-8')
            dataset = OCRDataset(folder, DeepTokenizer(verbose=False), target_height=32, patch_size=32,
                                 patch_size_w=8, use_2d_structure=False,
                                 conventions=ConventionTable(['expand_abbreviations'], {}, flip_prob=1.0))
            # canonicalize decomposes umlauts, as it does for the GT
            self.assertEqual(dataset.data_pairs[0][1:],
                             ('Gschf.: Haus', {'expand_abbreviations': 'Gescha\u0308ftsfu\u0308hrer: Haus'}))
            self.assertEqual(dataset[0][3], [1.0])

    def test_the_dataset_reads_the_line_read_from_its_image(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            Image.new('RGB', (400, 32), 'white').save(folder / 'line.png')
            (folder / 'line.gt.txt').write_text('Wasser', encoding='utf-8')
            (folder / 'line.diplomatic.txt').write_text('Waſſer', encoding='utf-8')
            dataset = OCRDataset(folder, DeepTokenizer(verbose=False), target_height=32, patch_size=32,
                                 patch_size_w=8, use_2d_structure=False,
                                 conventions=ConventionTable(['long_s'], {}, flip_prob=0.0))
            self.assertEqual(dataset.data_pairs[0][1:], ('Wasser', {'diplomatic': 'Waſſer'}))


class RemapColumns(unittest.TestCase):
    def test_columns_follow_their_names(self):
        weight = torch.tensor([[3.0, 0.0], [4.0, 6.0]])
        remapped = remap_convention_columns(weight, ['long_s', 'oblique_hyphen'],
                                            ['oblique_hyphen', 'abbreviations', 'long_s'])
        self.assertTrue(torch.equal(remapped[:, [0, 2]], torch.tensor([[0.0, 3.0], [6.0, 4.0]])))

    def test_a_new_flag_starts_as_long_as_the_trained_ones(self):
        weight = torch.tensor([[3.0, 0.0], [4.0, 6.0]])
        remapped = remap_convention_columns(weight, ['long_s', 'oblique_hyphen'],
                                            ['oblique_hyphen', 'abbreviations', 'long_s'])
        self.assertAlmostEqual(remapped[:, 1].norm().item(), 5.5, places=5)

    def test_without_trained_flags_a_new_one_starts_at_zero(self):
        remapped = remap_convention_columns(torch.zeros(2, 0), [], ['abbreviations'])
        self.assertTrue(torch.equal(remapped, torch.zeros(2, 1)))


class TrainedFlagRate(unittest.TestCase):
    def step(self, new_names):
        projection = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.zeros_(projection.weight)
        encoder_weight = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([{'params': [encoder_weight], 'lr': 0.1},
                                     {'params': [projection.weight], 'lr': 1.0}])
        hold_trained_flags_to_rate(optimizer, projection.weight, ['long_s', 'abbreviations'],
                                   lambda: new_names, slow_group=optimizer.param_groups[0],
                                   fast_group=optimizer.param_groups[1])
        projection.weight.grad = torch.ones_like(projection.weight)
        optimizer.step()
        return projection.weight.detach()

    def test_a_trained_flag_moves_at_the_encoder_rate_and_a_new_one_at_the_flag_rate(self):
        self.assertTrue(torch.allclose(self.step(['abbreviations']), torch.tensor([[-0.1, -1.0]])))

    def test_without_new_flags_every_flag_moves_at_the_encoder_rate(self):
        self.assertTrue(torch.allclose(self.step([]), torch.tensor([[-0.1, -0.1]])))


class CollateFlags(unittest.TestCase):
    def sample(self, flags):
        return torch.zeros(3, 32, 16), torch.tensor([5, 6]), 'label', flags

    def test_per_sample_flags_are_stacked(self):
        batch = Collate(patch_size=32, patch_size_w=8)([self.sample([1.0, 0.0]), self.sample([0.0, 1.0])])
        self.assertTrue(torch.equal(batch[-1], torch.tensor([[1.0, 0.0], [0.0, 1.0]])))
        self.assertEqual(batch[-2], ['label', 'label'])

    def test_a_run_without_conventions_gets_none(self):
        batch = Collate(patch_size=32, patch_size_w=8)([self.sample(None)])
        self.assertIsNone(batch[-1])
        self.assertEqual(batch[-2], ['label'])


os.environ.setdefault('OCR_PAIR_CACHE', '0')

if __name__ == '__main__':
    unittest.main()
