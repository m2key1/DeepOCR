import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import torch
from torch.optim.lr_scheduler import LambdaLR

from tests.test_grad_accumulation import batches_of, lines, tiny_model
from utils.loss import CTCLossWithSmoothing
from utils.model import normalize_images
from utils.tokenizer import DeepTokenizer


def poisoned(batch: tuple) -> tuple:
    images = normalize_images(batch[0]).clone()
    images[0, 0, 0, 0] = float('nan')
    return (images, *batch[1:])


def train(tokenizer: DeepTokenizer, batches: list, log_dir: Path) -> tuple[dict, dict, int]:
    model = tiny_model(tokenizer)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = LambdaLR(optimizer, lambda step: 1.0)
    with mock.patch.object(torch, 'autocast', lambda **kwargs: nullcontext()):
        model.train_model(batches, None, optimizer=optimizer,
                          criterion=CTCLossWithSmoothing(blank_id=tokenizer.blank_token_id),
                          epochs=1, log_dir=log_dir, scheduler=scheduler, router_warmup_epochs=0)
    checkpoint = torch.load(log_dir / 'last_checkpoint.pt', weights_only=False)
    return model.state_dict(), checkpoint['ema_state_dict'], scheduler.last_epoch


class NonFiniteStep(unittest.TestCase):
    def assert_same(self, state: dict, reference_state: dict):
        self.assertEqual(state.keys(), reference_state.keys())
        for key, value in reference_state.items():
            torch.testing.assert_close(state[key], value, msg=key)

    def test_a_batch_with_a_non_finite_gradient_leaves_no_trace(self):
        tokenizer = DeepTokenizer(normalize_unicode=True, verbose=False)
        clean, other = batches_of(lines(tokenizer, 4), [2, 2])
        with tempfile.TemporaryDirectory() as poisoned_dir, tempfile.TemporaryDirectory() as clean_dir:
            weights, ema, steps = train(tokenizer, [poisoned(other), clean], Path(poisoned_dir))
            reference_weights, reference_ema, reference_steps = train(tokenizer, [clean], Path(clean_dir))
        self.assertEqual((steps, reference_steps), (1, 1))
        self.assert_same(weights, reference_weights)
        self.assert_same(ema, reference_ema)


if __name__ == '__main__':
    unittest.main()
