import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from tests.test_token_decoder import MEMORY_DIM, save_tiny_lm
from utils.model import DeepOCR
from utils.token_decoder import END_TOKEN_ID, PAD_TOKEN_ID, TokenDecoder
from utils.tokenizer import DeepTokenizer


def teacher_forced_score(decoder, memory, padding, sequence):
    """Summed log probability of one generated row, up to and including its end token."""
    tokens = sequence[1:].tolist()
    length = tokens.index(END_TOKEN_ID) + 1 if END_TOKEN_ID in tokens else len(tokens)
    with torch.no_grad():
        logits = decoder(memory, padding, sequence[None, :length])
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    return sum(log_probs[step, token].item() for step, token in enumerate(tokens[:length]))


class BeamSearch(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        save_tiny_lm(Path(self.folder.name))
        self.decoder = TokenDecoder.from_pretrained_lm(Path(self.folder.name), MEMORY_DIM).eval()
        torch.manual_seed(1)
        for layer in self.decoder.layers:
            torch.nn.init.normal_(layer.cross_attn.o_proj.weight)
        self.memory = torch.randn(7, 2, MEMORY_DIM)
        self.padding = torch.tensor([[False] * 7, [False] * 5 + [True] * 2])

    def tearDown(self):
        self.folder.cleanup()

    def test_a_single_beam_is_greedy_decoding(self):
        greedy = self.decoder.generate(self.memory, self.padding, max_len=6)
        sequences, _ = self.decoder.beam_search(self.memory, self.padding, max_len=6, beam=1)
        self.assertTrue(torch.equal(sequences[:, 0], greedy))

    def test_beams_come_best_first_and_score_their_own_tokens(self):
        sequences, scores = self.decoder.beam_search(self.memory, self.padding, max_len=6, beam=3)
        self.assertEqual(tuple(sequences.shape[:2]), (2, 3))
        for row in range(2):
            self.assertEqual(scores[row].tolist(), sorted(scores[row].tolist(), reverse=True))
            for beam in range(3):
                expected = teacher_forced_score(self.decoder, self.memory[:, row:row + 1],
                                                self.padding[row:row + 1], sequences[row, beam])
                self.assertAlmostEqual(scores[row, beam].item(), expected, places=4)

    def test_the_best_beam_scores_at_least_the_greedy_line(self):
        greedy = self.decoder.generate(self.memory, self.padding, max_len=6)
        _, scores = self.decoder.beam_search(self.memory, self.padding, max_len=6, beam=4)
        for row in range(2):
            greedy_score = teacher_forced_score(self.decoder, self.memory[:, row:row + 1],
                                                self.padding[row:row + 1], greedy[row])
            self.assertGreaterEqual(scores[row, 0].item(), greedy_score - 1e-4)

    def test_no_beam_passes_its_byte_budget(self):
        free, _ = self.decoder.beam_search(self.memory, None, max_len=6, beam=3)
        longest = max(int(self.decoder.token_bytes[free[0, beam]].sum()) for beam in range(3))
        self.assertGreater(longest, 0)
        capped, _ = self.decoder.beam_search(self.memory, None, max_len=6, beam=3,
                                             max_bytes=torch.tensor([longest - 1, 1000]))
        for beam in range(3):
            self.assertLessEqual(int(self.decoder.token_bytes[capped[0, beam]].sum()), longest - 1)
        self.assertTrue(torch.equal(capped[1], free[1]))

    def test_finished_beams_carry_padding_after_their_end_token(self):
        sequences, _ = self.decoder.beam_search(self.memory, self.padding, max_len=6, beam=3)
        for row in sequences.reshape(-1, sequences.size(-1)).tolist():
            tokens = row[1:]
            if END_TOKEN_ID in tokens:
                after = tokens[tokens.index(END_TOKEN_ID) + 1:]
                self.assertEqual(set(after) | {PAD_TOKEN_ID}, {PAD_TOKEN_ID})


class JointChoice(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        save_tiny_lm(Path(self.folder.name))
        torch.manual_seed(0)
        self.tokenizer = DeepTokenizer(normalize_unicode=True, verbose=False)
        self.model = DeepOCR(tokenizer=self.tokenizer, backbone_name='cnn_grouped', cnn_width_stride=8,
                             cnn_patch_size=32, target_height=64, use_2d_structure=False, num_experts=4,
                             top_k_experts=2, expert_ffn='swiglu', expert_hidden=32, router_balance='loss_free',
                             num_encoder_layers=1, use_decoder=True, decoder_type='token',
                             token_decoder_lm_path=Path(self.folder.name)).eval()
        for layer in self.model.decoder.layers:
            torch.nn.init.normal_(layer.cross_attn.o_proj.weight)
        with torch.no_grad():
            self.memory, _, _, _ = self.model._encode(torch.rand(2, 3, 64, 256))
        self.padding = torch.zeros(2, self.memory.size(0), dtype=torch.bool)

    def tearDown(self):
        self.folder.cleanup()

    def test_text_the_frames_cannot_hold_is_impossible_for_the_ctc(self):
        blank = self.tokenizer.blank_token_id
        log_probs = F.log_softmax(torch.randn(3, 2, self.model.num_classes), dim=-1)
        likelihood = self.model._ctc_log_likelihood(log_probs, torch.tensor([3, 3]), ['ab', 'abcd'])
        self.assertTrue(torch.isfinite(likelihood[0]))
        self.assertEqual(likelihood[1].item(), float('-inf'))
        empty = self.model._ctc_log_likelihood(log_probs, torch.tensor([3, 3]), ['', ''])
        self.assertAlmostEqual(empty[0].item(), log_probs[:, 0, blank].sum().item(), places=5)

    def test_without_the_ctc_the_best_beam_wins(self):
        sequences, _ = self.model.decoder.beam_search(
            self.memory, self.padding, max_len=8, beam=3,
            max_bytes=self.model._decoder_byte_budget(self.memory, self.padding))
        best = self.model._decoder_ids_to_text(sequences[:, 0])
        chosen = self.model.decode_joint(self.memory, self.padding, beam=3, ctc_weight=0.0, length_bonus=0.0,
                                         max_len=8)
        self.assertEqual(chosen, best)

    def test_a_dominant_ctc_weight_picks_the_beam_the_image_supports_best(self):
        sequences, _ = self.model.decoder.beam_search(
            self.memory, self.padding, max_len=8, beam=3,
            max_bytes=self.model._decoder_byte_budget(self.memory, self.padding))
        log_probs = F.log_softmax(self.model.output_layer(self.memory).float(), dim=2)
        lengths = (~self.padding).sum(dim=1)
        expected = []
        for row in range(2):
            texts = self.model._decoder_ids_to_text(sequences[row])
            likelihood = self.model._ctc_log_likelihood(log_probs[:, [row] * 3], lengths[[row] * 3], texts)
            expected.append(texts[int(likelihood.argmax())])
        chosen = self.model.decode_joint(self.memory, self.padding, beam=3, ctc_weight=1e6, length_bonus=0.0,
                                         max_len=8)
        self.assertEqual(chosen, expected)


if __name__ == '__main__':
    unittest.main()
