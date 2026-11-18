import unittest
from unittest import mock

import torch

from tests.test_grad_accumulation import tiny_model
from utils.tokenizer import DeepTokenizer


class Masking(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DeepTokenizer(normalize_unicode=True, verbose=False)
        self.model = tiny_model(self.tokenizer)
        self.model.use_span_mask = True
        self.model.set_mask_ratio(0.5)
        self.model.pos_encoder.dropout.p = 0.0
        self.model.train()
        torch.manual_seed(1)
        self.images = torch.rand(2, 3, 64, 160)

    def encode(self):
        torch.manual_seed(2)
        _, _, _, state = self.model._encode(self.images)
        mask = state['active_span_mask'][0]
        self.assertGreater(int(mask.sum()), 1)
        return mask, state['projected_sequence'][:, 0]

    def test_by_default_every_masked_column_is_the_same_vector(self):
        mask, sequence = self.encode()
        torch.testing.assert_close(sequence[mask], self.model.mask_token[0].expand(int(mask.sum()), -1))

    def test_a_masked_column_can_keep_its_position(self):
        self.model.mask_keeps_position = True
        mask, sequence = self.encode()
        placed = self.model.mask_token[0, 0] + self.model.pos_encoder.pe[:len(mask), 0]
        torch.testing.assert_close(sequence[mask], placed[mask])

    def decoder_memory(self):
        targets = [self.tokenizer.encode(text) for text in ('abc', 'de')]
        decoder_input, _, padding = self.model._build_decoder_io(
            torch.tensor([token for target in targets for token in target]),
            torch.tensor([len(target) for target in targets]))
        captured = {}
        teacher_forced = self.model._decode_teacher_forced

        def capture(memory, *args):
            captured['memory'] = memory
            return teacher_forced(memory, *args)

        torch.manual_seed(2)
        with mock.patch.object(self.model, '_decode_teacher_forced', capture):
            self.model(self.images, decoder_targets=decoder_input, decoder_tgt_key_padding_mask=padding)
        self.model.use_span_mask = False
        unmasked, *_ = self.model._encode(self.images)
        return captured['memory'], unmasked

    def test_by_default_the_decoder_reads_the_masked_encoder(self):
        memory, unmasked = self.decoder_memory()
        self.assertFalse(torch.allclose(memory, unmasked))

    def test_the_decoder_can_read_an_unmasked_pass_of_the_encoder(self):
        self.model.decoder_reads_unmasked = True
        memory, unmasked = self.decoder_memory()
        torch.testing.assert_close(memory, unmasked)


if __name__ == '__main__':
    unittest.main()
