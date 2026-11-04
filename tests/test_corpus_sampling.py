import collections
import unittest

from utils.dataset import WidthBucketedBatchSampler, corpus_epoch_indices

LABELS = ['a'] * 100 + ['b'] * 4 + ['c'] * 16


def count(indices):
    return dict(collections.Counter(LABELS[i] for i in indices))


class CorpusShares(unittest.TestCase):
    def test_corpora_are_drawn_by_the_square_root_of_their_size(self):
        indices = corpus_epoch_indices(LABELS, alpha=0.5, max_repeat=10, epoch=0, seed=0)
        self.assertEqual(count(indices), {'a': 75, 'b': 15, 'c': 30})

    def test_a_small_corpus_repeats_at_most_max_repeat_times(self):
        indices = corpus_epoch_indices(LABELS, alpha=0.5, max_repeat=3, epoch=0, seed=0)
        self.assertEqual(count(indices), {'a': 75, 'b': 12, 'c': 30})

    def test_alpha_one_keeps_the_natural_mix(self):
        indices = corpus_epoch_indices(LABELS, alpha=1.0, max_repeat=3, epoch=0, seed=0)
        self.assertEqual(sorted(indices), list(range(120)))

    def test_a_subsampled_corpus_shows_new_lines_in_the_next_epoch(self):
        first, second = (set(i for i in corpus_epoch_indices(LABELS, 0.5, 10, epoch, 0) if LABELS[i] == 'a')
                         for epoch in (0, 1))
        self.assertEqual(len(first), 75)
        self.assertEqual(first | second, set(range(100)))

    def test_the_draw_is_the_same_for_the_same_epoch_and_seed(self):
        self.assertEqual(corpus_epoch_indices(LABELS, 0.5, 3, 4, 7), corpus_epoch_indices(LABELS, 0.5, 3, 4, 7))
        self.assertNotEqual(corpus_epoch_indices(LABELS, 0.5, 3, 4, 7), corpus_epoch_indices(LABELS, 0.5, 3, 5, 7))


class SamplerUsesTheShares(unittest.TestCase):
    def test_one_epoch_of_batches_holds_the_drawn_lines(self):
        sampler = WidthBucketedBatchSampler(list(range(120)), batch_size=7, seed=0, labels=LABELS,
                                            sampling_alpha=0.5, max_repeat=3)
        indices = [i for batch in sampler for i in batch]
        self.assertEqual(count(indices), {'a': 75, 'b': 12, 'c': 30})
        self.assertEqual(len(sampler), 17)

    def test_without_labels_every_line_is_read_once(self):
        sampler = WidthBucketedBatchSampler(list(range(120)), batch_size=7, seed=0)
        self.assertEqual(sorted(i for batch in sampler for i in batch), list(range(120)))


if __name__ == '__main__':
    unittest.main()
