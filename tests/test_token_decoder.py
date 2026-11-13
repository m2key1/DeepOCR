import importlib
import os
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Olmo3Config, Olmo3ForCausalLM

import train
from utils.model import DeepOCR
from utils.token_decoder import END_TOKEN_ID, PAD_TOKEN_ID, TokenDecoder
from utils.tokenizer import DeepTokenizer

VOCAB = {'<|endoftext|>': 0, '<|padding|>': 1, 'Haus': 2, 'Burger': 3, 'vnſer': 4, 'Bürger': 5}
MEMORY_DIM = 16


def save_tiny_lm(folder: Path):
    torch.manual_seed(0)
    config = Olmo3Config(vocab_size=len(VOCAB), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                         sliding_window=64, tie_word_embeddings=True, eos_token_id=0, pad_token_id=1)
    model = Olmo3ForCausalLM(config).eval()
    # default init sets every norm weight to one, which would hide a norm loaded into the wrong place
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.5)
    model.save_pretrained(folder)
    tokenizer = Tokenizer(models.WordLevel(VOCAB, unk_token='<|padding|>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(['<|endoftext|>', '<|padding|>'])
    tokenizer.save(str(folder / 'tokenizer.json'))
    return model


class MatchesTheReference(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.reference = save_tiny_lm(Path(self.folder.name))
        self.decoder = TokenDecoder.from_pretrained_lm(Path(self.folder.name), MEMORY_DIM).eval()
        self.memory = torch.randn(7, 2, MEMORY_DIM)

    def tearDown(self):
        self.folder.cleanup()

    def test_before_training_the_decoder_is_the_pretrained_lm(self):
        input_ids = torch.tensor([[0, 2, 4, 3], [0, 5, 2, 2]])
        with torch.no_grad():
            expected = self.reference(input_ids=input_ids).logits
            actual = self.decoder(self.memory, None, input_ids)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_cached_greedy_decoding_follows_the_teacher_forced_argmax(self):
        for layer in self.decoder.layers:
            torch.nn.init.normal_(layer.cross_attn.o_proj.weight)
        padding = torch.tensor([[False] * 7, [False] * 5 + [True] * 2])
        generated = self.decoder.generate(self.memory, padding, max_len=5)
        with torch.no_grad():
            logits = self.decoder(self.memory, padding, generated)
        for row in range(2):
            produced = generated[row, 1:]
            steps = int((produced != PAD_TOKEN_ID).sum())
            self.assertTrue(torch.equal(logits[row, :steps].argmax(-1), produced[:steps]))

    def test_a_row_ends_where_its_next_token_would_pass_its_byte_budget(self):
        for layer in self.decoder.layers:
            torch.nn.init.normal_(layer.cross_attn.o_proj.weight)
        free = self.decoder.generate(self.memory, None, max_len=6)
        first_two = int(self.decoder.token_bytes[free[0, 1:3]].sum())
        capped = self.decoder.generate(self.memory, None, max_len=6, max_bytes=torch.tensor([first_two, 1000]))
        self.assertEqual(capped[0, :4].tolist(), free[0, :3].tolist() + [END_TOKEN_ID])
        self.assertTrue(torch.equal(capped[1], free[1]))

    def test_byte_lengths_count_text_tokens_and_skip_specials(self):
        self.assertEqual(self.decoder.token_bytes.tolist(), [0, 0, 4, 6, 5, 6])


class InputsAndTargets(unittest.TestCase):
    def test_lines_start_after_eos_and_end_with_it(self):
        with tempfile.TemporaryDirectory() as folder:
            save_tiny_lm(Path(folder))
            decoder = TokenDecoder.from_pretrained_lm(Path(folder), MEMORY_DIM)
        inputs, targets, padding = decoder.build_io(['vnſer Burger', 'Haus'], torch.device('cpu'))
        self.assertEqual(inputs.tolist(), [[0, 4, 3], [0, 2, 1]])
        self.assertEqual(targets.tolist(), [[4, 3, 0], [2, 0, 1]])
        self.assertEqual(padding.tolist(), [[False, False, False], [False, False, True]])

    def test_corruption_swaps_text_tokens_and_keeps_the_start_and_the_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            save_tiny_lm(Path(folder))
            decoder = TokenDecoder.from_pretrained_lm(Path(folder), MEMORY_DIM)
        inputs, _, padding = decoder.build_io(['vnſer Burger Haus', 'Haus'], torch.device('cpu'))
        self.assertTrue(torch.equal(decoder.corrupt_inputs(inputs, padding, 0.0), inputs))
        corrupted = decoder.corrupt_inputs(inputs, padding, 1.0)
        self.assertEqual(corrupted[:, 0].tolist(), [0, 0])
        self.assertEqual(corrupted[1, 2:].tolist(), [1, 1])
        text = ~padding
        text[:, 0] = False
        self.assertTrue(bool((corrupted[text] != inputs[text]).all()))
        self.assertTrue(bool(((corrupted[text] >= 2) & (corrupted[text] < len(VOCAB))).all()))

    def test_text_is_encoded_composed(self):
        with tempfile.TemporaryDirectory() as folder:
            save_tiny_lm(Path(folder))
            decoder = TokenDecoder.from_pretrained_lm(Path(folder), MEMORY_DIM)
        self.assertEqual(decoder.encode_texts(['Bürger']), [[5]])
        self.assertEqual(decoder.ids_to_text(torch.tensor([[0, 5, 2, 0, 1]])), ['Bürger Haus'])


class FrozenEncoder(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        save_tiny_lm(Path(self.folder.name))
        torch.manual_seed(0)
        self.tokenizer = DeepTokenizer(normalize_unicode=True, verbose=False)
        self.model = DeepOCR(tokenizer=self.tokenizer, backbone_name='cnn_grouped', cnn_width_stride=8,
                             cnn_patch_size=32, target_height=64, use_2d_structure=False, num_experts=4,
                             top_k_experts=2, expert_ffn='swiglu', expert_hidden=32, router_balance='loss_free',
                             num_encoder_layers=1, drop_path=0.5, mask_ratio=0.5, use_decoder=True,
                             decoder_label_smoothing=0.0, decoder_type='token',
                             token_decoder_lm_path=Path(self.folder.name))
        self.model.freeze_encoder()
        self.model.train()

    def tearDown(self):
        self.folder.cleanup()

    def train_one_step(self):
        images = torch.rand(2, 3, 64, 128)
        targets = [self.tokenizer.encode('Haus Burger'), self.tokenizer.encode('Haus')]
        target_lengths = torch.tensor([len(target) for target in targets])
        decoder_input, decoder_target, padding = self.model._build_decoder_io(
            torch.tensor([token for target in targets for token in target]), target_lengths)
        optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad], lr=1.0)
        _, _, _, logits = self.model(images, decoder_targets=decoder_input,
                                     decoder_tgt_key_padding_mask=padding)
        loss = torch.nn.functional.cross_entropy(logits.permute(1, 2, 0), decoder_target,
                                                 ignore_index=PAD_TOKEN_ID)
        loss.backward()
        optimizer.step()
        self.model._update_router_biases()

    def test_only_the_decoder_trains(self):
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        self.train_one_step()
        after = self.model.state_dict()
        changed = {name for name in before if not torch.equal(before[name], after[name])}
        self.assertIn('decoder.layers.0.cross_attn.o_proj.weight', changed)
        self.assertIn('decoder.embed_tokens.weight', changed)
        self.assertEqual({name for name in changed if not name.startswith('decoder')}, set())

    def test_the_encoder_runs_as_at_inference(self):
        self.assertTrue(self.model.decoder.training)
        self.assertFalse(self.model.backbone.training)
        self.assertFalse(self.model.transformer_layers.training)
        images = torch.rand(2, 3, 64, 128)
        with torch.no_grad():
            first = self.model(images)[0]
            second = self.model(images)[0]
        self.assertTrue(torch.equal(first, second))

    def test_the_byte_budget_follows_the_ctc_best_path(self):
        blank = self.tokenizer.blank_token_id
        best_path = torch.tensor([[blank, 5, 5, blank, 5, 7], [9, 9, blank, 9, blank, 9]])
        logits = torch.nn.functional.one_hot(best_path, self.model.num_classes).float().transpose(0, 1)
        self.model.output_layer = torch.nn.Identity()
        padding = torch.tensor([[False] * 6, [False] * 4 + [True] * 2])
        budget = self.model._decoder_byte_budget(logits, padding)
        self.assertEqual(budget.tolist(), [4 + 8, 3 + 8])

    def test_inputs_are_corrupted_only_while_training(self):
        self.model.decoder_input_noise = 1.0
        target = torch.tensor(self.tokenizer.encode('Haus Burger'))
        lengths = torch.tensor([len(target)])
        clean = self.model.decoder.build_io(['Haus Burger'], torch.device('cpu'))[0]
        self.assertFalse(torch.equal(self.model._build_decoder_io(target, lengths)[0], clean))
        self.model.eval()
        self.assertTrue(torch.equal(self.model._build_decoder_io(target, lengths)[0], clean))


if __name__ == '__main__':
    unittest.main()


class FineTuneKeepsTheBaseDecoder(unittest.TestCase):
    """A run warm-started from a checkpoint with a token decoder would build a character decoder,
    and the base's decoder weights would be skipped as belonging to another kind."""
    CONFIG = {'hidden_size': 32, 'vocab_size': len(VOCAB)}

    def configure(self, base: dict, **env) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'base.pt'
            torch.save({'model_state_dict': {}, **base}, path)
            saved = {name: os.environ.get(name) for name in ('TRAIN_BASE_CHECKPOINT', *env)}
            os.environ.update({'TRAIN_BASE_CHECKPOINT': str(path), **env})
            try:
                importlib.reload(train)
            finally:
                for name, value in saved.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
        self.addCleanup(importlib.reload, train)

    def test_a_token_decoder_base_gives_its_type_config_and_tokenizer(self):
        self.configure({'decoder_type': 'token', 'token_decoder_config': self.CONFIG,
                        'token_decoder_tokenizer': 'tokenizer json'})
        self.assertEqual(train.DECODER_TYPE, 'token')
        self.assertEqual(train.TOKEN_DECODER_CONFIG, self.CONFIG)
        self.assertEqual(train.TOKEN_DECODER_TOKENIZER, 'tokenizer json')

    def test_a_character_decoder_base_keeps_the_character_decoder(self):
        self.configure({'decoder_type': 'char'})
        self.assertEqual(train.DECODER_TYPE, 'char')
        self.assertIsNone(train.TOKEN_DECODER_CONFIG)

    def test_a_token_decoder_base_fine_tunes_with_the_settings_it_was_trained_with(self):
        self.configure({'decoder_type': 'token', 'token_decoder_config': self.CONFIG,
                        'token_decoder_tokenizer': 'tokenizer json'})
        self.assertEqual(train.TOKEN_DECODER_NEW_LR, 1e-4)
        self.assertIs(train.SELECT_ON_DECODER, True)
        self.assertEqual(train.DECODER_LABEL_SMOOTHING, 0.0)

    def test_a_character_decoder_base_keeps_the_usual_settings(self):
        self.configure({'decoder_type': 'char'})
        self.assertEqual(train.TOKEN_DECODER_NEW_LR, 3e-4)
        self.assertIs(train.SELECT_ON_DECODER, False)
        self.assertEqual(train.DECODER_LABEL_SMOOTHING, 0.1)

    def test_settings_the_run_names_win_over_the_base_defaults(self):
        self.configure({'decoder_type': 'token', 'token_decoder_config': self.CONFIG,
                        'token_decoder_tokenizer': 'tokenizer json'},
                       TRAIN_TOKEN_DECODER_NEW_LR='2e-4', TRAIN_SELECT_ON_DECODER='0')
        self.assertEqual(train.TOKEN_DECODER_NEW_LR, 2e-4)
        self.assertIs(train.SELECT_ON_DECODER, False)
        self.assertEqual(train.DECODER_LABEL_SMOOTHING, 0.0)

    def test_a_decoder_the_run_names_wins_over_the_base(self):
        self.configure({'decoder_type': 'token', 'token_decoder_config': self.CONFIG,
                        'token_decoder_tokenizer': 'tokenizer json'}, TRAIN_DECODER_TYPE='char')
        self.assertEqual(train.DECODER_TYPE, 'char')
        self.assertIsNone(train.TOKEN_DECODER_CONFIG)
