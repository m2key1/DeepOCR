import unittest

import train


class TrainingDefaults(unittest.TestCase):
    def test_the_decoder_never_learns_to_write_over_hidden_columns(self):
        self.assertIs(train.MASK_KEEPS_POSITION, True)
        self.assertIs(train.DECODER_READS_UNMASKED, True)
        self.assertEqual(train.NEGATIVE_LINE_PROB, 0.03)

    def test_a_short_run_saves_its_own_weights(self):
        self.assertIs(train.EMA_WARMUP, True)

    def test_the_ctc_head_trains_without_an_entropy_bonus(self):
        self.assertEqual(train.CTC_ENTROPY_WEIGHT, 0.0)


if __name__ == '__main__':
    unittest.main()
