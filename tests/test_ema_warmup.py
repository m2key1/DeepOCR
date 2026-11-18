import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import torch
from torch.optim.lr_scheduler import LambdaLR

from tests.test_grad_accumulation import batches_of, lines, tiny_model
from utils.loss import CTCLossWithSmoothing
from utils.model import ema_with_warmup
from utils.tokenizer import DeepTokenizer


def train(ema_warmup: bool, folder: Path) -> tuple[dict, dict, dict]:
    tokenizer = DeepTokenizer(normalize_unicode=True, verbose=False)
    model = tiny_model(tokenizer)
    start = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    with mock.patch.object(torch, 'autocast', lambda **kwargs: nullcontext()):
        model.train_model(batches_of(lines(tokenizer, 8), [1] * 8), None, optimizer=optimizer,
                          criterion=CTCLossWithSmoothing(blank_id=tokenizer.blank_token_id), epochs=1,
                          log_dir=folder, scheduler=LambdaLR(optimizer, lambda step: 1.0),
                          router_warmup_epochs=0, ema_warmup=ema_warmup)
    ema = torch.load(folder / 'last_checkpoint.pt', weights_only=False)['ema_state_dict']
    return start, model.state_dict(), {key.removeprefix('module.'): value for key, value in ema.items()}


def distance(first: dict, second: dict, key: str) -> float:
    return float((first[key] - second[key]).norm())


class EmaWarmup(unittest.TestCase):
    def test_the_decay_grows_with_the_updates(self):
        averaged, integer = [torch.zeros(3)], [torch.zeros(1, dtype=torch.long)]
        update = ema_with_warmup(0.999)
        for count in (1, 2, 3):
            update(averaged, [torch.ones(3)], torch.tensor(count))
            update(integer, [torch.tensor([7])], torch.tensor(count))
        # the start keeps 2/11 * 3/12 * 4/13 of the average
        torch.testing.assert_close(averaged[0], torch.full((3,), 1 - 24 / 1716))
        self.assertEqual(integer[0].tolist(), [7])

    def test_a_short_run_saves_weights_near_its_end_rather_than_its_start(self):
        key = 'output_layer.weight'
        with tempfile.TemporaryDirectory() as fixed_dir, tempfile.TemporaryDirectory() as warm_dir:
            start, end, fixed = train(False, Path(fixed_dir))
            _, warm_end, warm = train(True, Path(warm_dir))
        torch.testing.assert_close(warm_end[key], end[key])
        self.assertLess(distance(fixed, start, key), distance(fixed, end, key))
        self.assertLess(distance(warm, end, key), distance(warm, start, key))


if __name__ == '__main__':
    unittest.main()
