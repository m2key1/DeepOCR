import os
import socket
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from utils.dataset import Collate
from utils.loss import CTCLossWithSmoothing
from utils.model import DeepOCR, MoEBlock
from utils.tokenizer import DeepTokenizer

HEIGHT, WIDTH = 64, 96
TEXTS = ('abcdefgh', 'hgfedcba', 'bcdefgha', 'ahgfedcb', 'cdefghab', 'bahgfedc', 'defghabc', 'cbahgfed')
# each window takes two lines from each rank, so its one step covers all eight
RANK_LINES = ((0, 1, 4, 5), (2, 3, 6, 7))


def tiny_model(tokenizer: DeepTokenizer) -> DeepOCR:
    torch.manual_seed(0)
    model = DeepOCR(tokenizer=tokenizer, backbone_name='cnn_grouped', cnn_width_stride=8, cnn_patch_size=32,
                    target_height=HEIGHT, use_2d_structure=False, num_experts=4, top_k_experts=2,
                    expert_ffn='swiglu', expert_hidden=32, router_balance='loss_free', num_encoder_layers=1,
                    drop_path=0.0, use_decoder=True, decoder_dropout=0.0)
    # training-mode BatchNorm normalises by the batch's own statistics, and span masks, router
    # jitter and dropout draw from the generator, so each would tell a batch of four from two of
    # two apart by itself
    model.use_span_mask = False
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.BatchNorm2d):
            parent_name, _, child_name = name.rpartition('.')
            setattr(model.get_submodule(parent_name), child_name, nn.Identity())
        elif isinstance(module, nn.Dropout):
            module.p = 0.0
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.0
        elif isinstance(module, MoEBlock):
            module.jitter_noise = 0.0
    return model


def lines(tokenizer: DeepTokenizer, count: int) -> list:
    generator = torch.Generator().manual_seed(1)
    return [(torch.randint(0, 256, (3, HEIGHT, WIDTH), dtype=torch.uint8, generator=generator),
             tokenizer.encode(text), 'test', None) for text in TEXTS[:count]]


def batches_of(items: list, batch_sizes: list[int]) -> list:
    collate = Collate(patch_size=32, patch_size_w=8, use_2d_structure=False, fill_value=255)
    batches, start = [], 0
    for size in batch_sizes:
        batches.append(collate(items[start:start + size]))
        start += size
    return batches


def train(model: DeepOCR, tokenizer: DeepTokenizer, batches: list, grad_accum_steps: int, **distributed) -> LambdaLR:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = LambdaLR(optimizer, lambda step: 1.0)
    # bf16 rounds a sum over four lines differently from two sums over two
    with mock.patch.object(torch, 'autocast', lambda **kwargs: nullcontext()):
        model.train_model(batches, None, optimizer=optimizer,
                          criterion=CTCLossWithSmoothing(blank_id=tokenizer.blank_token_id),
                          epochs=1, scheduler=scheduler, router_warmup_epochs=0, grad_accum_steps=grad_accum_steps,
                          **distributed)
    return scheduler


def train_one_epoch(batch_sizes: list[int], grad_accum_steps: int) -> tuple[DeepOCR, LambdaLR]:
    tokenizer = DeepTokenizer(normalize_unicode=True, verbose=False)
    model = tiny_model(tokenizer)
    batches = batches_of(lines(tokenizer, sum(batch_sizes)), batch_sizes)
    return model, train(model, tokenizer, batches, grad_accum_steps)


def train_rank_of_two(rank: int, port: int, result: str):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    dist.init_process_group('gloo', rank=rank, world_size=2)
    tokenizer = DeepTokenizer(normalize_unicode=True, verbose=False)
    model = tiny_model(tokenizer)
    items = lines(tokenizer, len(TEXTS))
    batches = batches_of([items[index] for index in RANK_LINES[rank]], [2, 2])
    # static_graph as train.py builds it
    scheduler = train(model, tokenizer, batches, grad_accum_steps=2, ddp_model=DDP(model, static_graph=True),
                      is_master=rank == 0, world_size=2)
    if rank == 0:
        torch.save({'state': model.state_dict(), 'steps': scheduler.last_epoch}, result)
    dist.destroy_process_group()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


class GradientAccumulation(unittest.TestCase):
    def assert_same_weights(self, state: dict, reference_state: dict):
        self.assertEqual(state.keys(), reference_state.keys())
        for key, value in reference_state.items():
            torch.testing.assert_close(state[key], value, msg=key)

    def test_two_halves_accumulated_step_like_the_whole_batch(self):
        whole, whole_scheduler = train_one_epoch([4, 4], grad_accum_steps=1)
        halves, halves_scheduler = train_one_epoch([2, 2, 2, 2], grad_accum_steps=2)
        self.assertEqual(whole_scheduler.last_epoch, 2)
        self.assertEqual(halves_scheduler.last_epoch, 2)
        self.assert_same_weights(halves.state_dict(), whole.state_dict())

    def test_a_short_last_window_steps_at_its_own_size(self):
        reference, reference_scheduler = train_one_epoch([4, 2], grad_accum_steps=1)
        accumulated, accumulated_scheduler = train_one_epoch([2, 2, 2], grad_accum_steps=2)
        self.assertEqual(reference_scheduler.last_epoch, 2)
        self.assertEqual(accumulated_scheduler.last_epoch, 2)
        self.assert_same_weights(accumulated.state_dict(), reference.state_dict())

    def test_two_ranks_accumulating_step_like_one_process_with_every_line(self):
        whole, whole_scheduler = train_one_epoch([8], grad_accum_steps=1)
        with tempfile.TemporaryDirectory() as folder:
            result = Path(folder) / 'rank0.pt'
            torch.multiprocessing.spawn(train_rank_of_two, args=(free_port(), str(result)), nprocs=2)
            trained = torch.load(result)
        self.assertEqual(whole_scheduler.last_epoch, 1)
        self.assertEqual(trained['steps'], 1)
        self.assert_same_weights(trained['state'], whole.state_dict())

    def test_the_router_load_counts_every_batch_until_the_step(self):
        torch.manual_seed(0)
        block = MoEBlock(d_model=16, num_experts=4, top_k=2, dim_feedforward=32, jitter_noise=0.0,
                         expert_ffn='swiglu', router_balance='loss_free').train()
        first, second = torch.randn(5, 2, 16), torch.randn(7, 2, 16)
        loads = []
        for tokens in (first, second):
            block(tokens)
            loads.append(block._router_load)
            block._router_load = None
        block(first)
        block(second)
        torch.testing.assert_close(block._router_load, loads[0] + loads[1])
        # 12 columns of 2 lines, each routed to 2 experts
        self.assertEqual(block._router_load.sum().item(), 48)


if __name__ == '__main__':
    unittest.main()
