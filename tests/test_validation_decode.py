import unittest
from contextlib import nullcontext
from unittest import mock

import torch
from torch.optim.lr_scheduler import LambdaLR

from tests.test_grad_accumulation import TEXTS, batches_of, lines, tiny_model
from utils.loss import CTCLossWithSmoothing
from utils.tokenizer import DeepTokenizer


class ValidationDecode(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DeepTokenizer(normalize_unicode=True, verbose=False)
        self.model = tiny_model(self.tokenizer)
        self.batches = batches_of(lines(self.tokenizer, 4), [2, 2])
        self.criterion = CTCLossWithSmoothing(blank_id=self.tokenizer.blank_token_id)

    def test_the_decoder_metric_scores_the_reading_inference_ships(self):
        shipped = iter([list(TEXTS[0:2]), list(TEXTS[2:4])])
        with mock.patch.object(self.model, 'decode_text', side_effect=lambda *args, **kwargs: next(shipped)):
            results = self.model.validate(self.criterion, self.batches, torch.device('cpu'), writer=None,
                                          epoch=None, run_decoder=True)
        self.assertEqual(results['decoder']['cer'], 0.0)

    def test_the_beam_validated_during_training_has_no_length_penalty(self):
        penalties = []
        beam_search = DeepTokenizer.batch_decode_beam_search

        def recording(tokenizer, *args, **kwargs):
            penalties.append(kwargs.get('length_penalty'))
            return beam_search(tokenizer, *args, **kwargs)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0)
        with mock.patch.object(DeepTokenizer, 'batch_decode_beam_search', recording), \
                mock.patch.object(torch, 'autocast', lambda **kwargs: nullcontext()):
            self.model.train_model(self.batches, self.batches, optimizer=optimizer, criterion=self.criterion,
                                   epochs=1, scheduler=LambdaLR(optimizer, lambda step: 1.0), beam_width=2,
                                   beam_val_frequency=1, router_warmup_epochs=0)
        self.assertEqual(penalties, [0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
