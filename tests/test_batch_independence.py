import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from utils.backbones import masked_norm
from utils.model import DeepOCR
from utils.tokenizer import DeepTokenizer

SHORT, LONG = 40, 200


def tiny_model(batch_independent: bool) -> DeepOCR:
    torch.manual_seed(0)
    model = DeepOCR(tokenizer=DeepTokenizer(normalize_unicode=True, verbose=False), backbone_name='cnn_grouped',
                    cnn_width_stride=8, cnn_patch_size=32, target_height=64, use_2d_structure=False, num_experts=4,
                    top_k_experts=2, expert_ffn='swiglu', expert_hidden=32, router_balance='loss_free',
                    num_encoder_layers=1, batch_independent_backbone=batch_independent)
    # a fresh BatchNorm is the identity and zero-initialised residual branches add nothing, which
    # would hide any leak; trained-looking statistics and weights expose it
    with torch.no_grad():
        for module in model.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.uniform_(0.5, 1.5)
                module.bias.uniform_(-0.5, 0.5)
                module.running_mean.uniform_(-0.5, 0.5)
                module.running_var.uniform_(0.5, 1.5)
    return model.eval()


def short_line_features(model: DeepOCR, beside_long: bool):
    """The encoder memory and CTC log-probs of the short line's own columns, read alone or padded
    white to the width of a long line it shares a batch with."""
    generator = torch.Generator().manual_seed(1)
    short = torch.randint(0, 256, (3, 64, SHORT), dtype=torch.uint8, generator=generator)
    long = torch.randint(0, 256, (3, 64, LONG), dtype=torch.uint8, generator=generator)
    if beside_long:
        padded = torch.full((3, 64, LONG), 255, dtype=torch.uint8)
        padded[:, :, :SHORT] = short
        images = torch.stack([padded, long])
        masks = torch.zeros(2, LONG // 8, dtype=torch.bool)
        masks[0, SHORT // 8:] = True
    else:
        images = short.unsqueeze(0)
        masks = torch.zeros(1, SHORT // 8, dtype=torch.bool)
    with torch.no_grad():
        memory, _, _, _ = model._encode(images, attention_masks=masks)
        log_probs = model.output_layer(memory).log_softmax(dim=-1)
    columns = SHORT // 8
    return memory[:columns, 0], log_probs[:columns, 0]


class ReadingDoesNotDependOnTheBatch(unittest.TestCase):
    def test_a_short_line_reads_the_same_alone_and_beside_a_long_one(self):
        model = tiny_model(batch_independent=True)
        alone_memory, alone_log_probs = short_line_features(model, beside_long=False)
        beside_memory, beside_log_probs = short_line_features(model, beside_long=True)
        torch.testing.assert_close(beside_memory, alone_memory, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(beside_log_probs, alone_log_probs, atol=1e-5, rtol=1e-5)

    def test_the_old_backbone_lets_the_padding_in(self):
        model = tiny_model(batch_independent=False)
        alone_memory, _ = short_line_features(model, beside_long=False)
        beside_memory, _ = short_line_features(model, beside_long=True)
        self.assertGreater((beside_memory - alone_memory).abs().max().item(), 1e-3)


class TrainingStatisticsSkipPadding(unittest.TestCase):
    def test_batch_statistics_come_from_the_lines_own_columns(self):
        torch.manual_seed(0)
        line = torch.randn(1, 4, 2, 5)
        padded = torch.cat([line, torch.full((1, 4, 2, 7), 7.0)], dim=3)
        keep = torch.zeros(1, 1, 1, 12)
        keep[..., :5] = 1.0
        alone, beside = nn.BatchNorm2d(4).train(), nn.BatchNorm2d(4).train()
        expected = alone(line)
        got = masked_norm(beside, padded, keep)
        torch.testing.assert_close(got[..., :5], expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(beside.running_mean, alone.running_mean, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(beside.running_var, alone.running_var, atol=1e-6, rtol=1e-6)


class MultiGpuTraining(unittest.TestCase):
    def test_a_backbone_converted_to_sync_batchnorm_reads_as_before(self):
        """Distributed training swaps every BatchNorm for SyncBatchNorm; the masking must survive it."""
        model = tiny_model(batch_independent=True)
        before, _ = short_line_features(model, beside_long=True)
        model.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(model.backbone)
        self.assertTrue(any(isinstance(m, nn.SyncBatchNorm) for m in model.backbone.modules()))
        after, _ = short_line_features(model, beside_long=True)
        torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-6)


class CheckpointsKeepTheirBackbone(unittest.TestCase):
    def test_the_setting_survives_a_save_and_an_older_checkpoint_reads_as_before(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pt'
            tiny_model(batch_independent=True).save(path)
            self.assertTrue(DeepOCR.load(str(path))[0].backbone.batch_independent)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            checkpoint.pop('batch_independent_backbone')
            torch.save(checkpoint, path)
            self.assertFalse(DeepOCR.load(str(path))[0].backbone.batch_independent)


if __name__ == '__main__':
    unittest.main()
