import random
import unittest

import torch

from utils.conventions import (ConventionTable, normalize_long_s, normalize_typographic_punctuation,
                               remap_convention_columns, restore_long_s_by_position,
                               restore_long_s_german)
from utils.dataset import Collate

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
        outcomes = {table(flip_prob=0.5).sample('diplomatic', 'Waſſer⸗', rng)[0] for _ in range(200)}
        self.assertEqual(outcomes, {'Waſſer⸗', 'Wasser⸗', 'Waſſer-', 'Wasser-'})

    def test_an_unknown_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "state 'yes' for 'fraktur'"):
            table({'long_s': {'fraktur': 'yes'}})

    def test_image_state_labels_without_samples_are_reported(self):
        states = table({'long_s': {'fraktur': 'german', 'frakture': 'german'}})
        self.assertEqual(states.unseen_labels({'fraktur', 'diplomatic'}), {'long_s': ['frakture']})


class RemapColumns(unittest.TestCase):
    def test_columns_follow_their_names_and_new_flags_start_at_zero(self):
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        remapped = remap_convention_columns(weight, ['long_s', 'oblique_hyphen'],
                                            ['oblique_hyphen', 'abbreviations', 'long_s'])
        self.assertTrue(torch.equal(remapped, torch.tensor([[2.0, 0.0, 1.0], [4.0, 0.0, 3.0]])))


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


if __name__ == '__main__':
    unittest.main()
