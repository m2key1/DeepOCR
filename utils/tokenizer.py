import itertools
import unicodedata
from pathlib import Path
from typing import Sequence

import torch

# Equivalences NFD does not cover: code points that render as the same pen stroke and so cannot
# be told apart from pixels. Sources differ in whether they keep the typographic form, so folding
# gives the same mark one target.
#
# Not merged: „ vs “ are a low and a raised mark, and « » are distinct shapes. Greek is real Greek
# text, so its Latin homoglyphs stay. Cyrillic occurs only as typos inside Latin words.
CANONICAL_MAP = {
    '´': "'",       # ACUTE ACCENT
    '`': "'",       # GRAVE ACCENT
    ' ': ' ',       # NO-BREAK SPACE
    'а': 'a', 'е': 'e', 'к': 'k',   # Cyrillic homoglyphs
    'л': 'l', 'м': 'm', 'р': 'p',

    'ꝛ': 'r',       # LATIN SMALL LETTER R ROTUNDA

    # Fraktur and gazette symbols from the German corpora (reichsanzeiger, mkn_kurrent,
    # UB-Mannheim), folded so they need no vocabulary entry.
    #
    # Typographic variants, no meaningful loss:
    '̅': '̄',    # COMBINING OVERLINE -> COMBINING MACRON, same nasal-abbreviation stroke
    'ʼ': "'", 'ˀ': "'",  # MODIFIER LETTER APOSTROPHE / GLOTTAL STOP -> apostrophe
    'ℛ': 'R', 'ℓ': 'l', 'ϑ': 'θ',  # SCRIPT R / SCRIPT l / THETA SYMBOL -> plain forms
    'ʃ': 's', 'ʝ': 'j',    # ESH / J-CROSSED-TAIL -> nearest plain letter
    '͠': '̃',    # COMBINING DOUBLE TILDE -> combining tilde
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',  # superscript digits -> plain (footnote markers)
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    '⅛': '1/8', '⅜': '3/8', '⅝': '5/8', '⅞': '7/8', '⅚': '5/6',  # vulgar fractions -> N/M
    '⅙': '1/6', '⅐': '1/7', '⅑': '1/9', '⅒': '1/10', '⅖': '2/5', '⅗': '3/5',
    '✳': '*',              # U+2733 EIGHT SPOKED ASTERISK -> plain asterisk (decorative variant)
    #
    # Meaningful symbols with no in-vocab token, folded lossily to avoid a vocab and KenLM change:
    'ℳ': 'M',              # U+2133 German MARK currency sign -> M [LOSSY: currency]
    'ℒ': 'L', 'ℬ': 'B', 'ℭ': 'C',  # other script-capital letters -> plain
    'ʒ': 'z', 'Ʒ': 'Z',    # EZH -> z [LOSSY: a distinct letter in Middle High German]
    '℔': 'lb',             # L-B BAR (pound weight) -> lb
    #
    # Encoding errors and non-textual marks are deleted, the line is kept:
    '�': '', '': '',        # REPLACEMENT CHAR / MUFI private-use junk
    '☚': '', '☛': '', '☉': '', '□': '',  # manicules, sun symbol, empty box
    '⟦': '', '⟧': '', '₰': '', '₤': '', 'ӕ': '',  # editorial brackets, penny/lira signs, cyrillic-error
}

# Not folded, because corpora disagree on them and CONVENTION_NAMES in train.py conditions on them
# instead. Anything added back here must also leave the CONVENTION_NAMES flag list.
#   ſ  U+017F LONG S            ⸗  U+2E17 DOUBLE OBLIQUE HYPHEN
#   ’ ‘ U+2019 U+2018           ”  U+201D           –  U+2013 EN DASH
#
# Not folded, though NFKD would collapse every one of them:
#   Ⓑ Ⓒ Ⓚ Ⓜ ㊵ ⑥⑦⑧⑨  circled markers that carry meaning in printed directory tables
#   ꝑ ꝓ ꝗ ꝙ ꝯ ꝝ ꝰ ẜ    scribal abbreviations; U+A770 'ꝰ' has NFKD 'ꝯ', one abbreviation
#                       decomposing into another, so blanket compatibility folding is unsafe
#   ᵃᶜᵈᵉⁱⁿᵒˢᵗᵘ          superscript modifiers: abbreviation devices, not allographs
#   ¹²³ ½ ¼ ¾          footnote markers and fractions, semantically real
_CANONICAL_TABLE = str.maketrans(CANONICAL_MAP)


def canonicalize(text: str) -> str:
    """Idempotent, so it runs both in prepare_data.py (keeping the LM corpus and CER references
    canonical) and again in encode() for data that bypassed it. An addition to CANONICAL_MAP makes
    its source token unreachable, so rebuild the KenLM dicts in the same change."""
    return unicodedata.normalize('NFD', text).translate(_CANONICAL_TABLE)


def check_kenlm_artifacts(kenlm_path, lm_dict_path, tokens) -> tuple:
    """Returns the pair to use, or (None, None) with the reason printed. A .dict built against
    another vocabulary decodes to plausible-looking text rather than failing, hence the length
    check. A stale LM of the right shape is not detectable; rebuild after changing CANONICAL_MAP."""
    if not kenlm_path:
        return None, None
    for label, path in (('KENLM_PATH', kenlm_path), ('LM_DICT_PATH', lm_dict_path)):
        if not path or not Path(path).exists():
            print(f"{label} {path} not found; decoding without a fused LM. Build one with build_kenlm.py.")
            return None, None
    entries = len(Path(lm_dict_path).read_text(encoding='utf-8').splitlines())
    if entries != len(tokens):
        print(f"LM dict {lm_dict_path} has {entries} entries for a {len(tokens)}-token inventory; "
              f"it is not index-parallel, so it is ignored. Rebuild it with build_kenlm.py "
              f"--checkpoint against the model it is used with.")
        return None, None
    return kenlm_path, lm_dict_path


class DeepTokenizer:

    def __init__(
        self,
        unk_token='[UNK]',
        pad_token='[PAD]',
        blank_token='[BLANK]',
        bos_token='[BOS]',
        eos_token='[EOS]',
        normalize_unicode=True,
        verbose=False,
        tokens: Sequence[str] | None = None,
        kenlm_path: str | Path | None = None,
        lm_dict_path: str | Path | None = None,
        lm_weight: float = 2.0,
    ):
        self.blank_token = blank_token
        self.blank_token_id = 0
        self.unk_token = unk_token
        self.unk_token_id = 1
        self.pad_token = pad_token
        self.pad_token_id = 2
        self.bos_token = bos_token
        self.bos_token_id = 3
        self.eos_token = eos_token
        self.eos_token_id = 4

        self.normalize_unicode = normalize_unicode
        self.verbose = verbose
        self.kenlm_path = kenlm_path
        self.lm_dict_path = lm_dict_path
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.special_tokens = [self.blank_token, self.unk_token, self.pad_token, self.bos_token, self.eos_token]

        if tokens is not None:
            # a restored checkpoint's inventory must match its output head shape
            for idx, tok in enumerate(tokens):
                self.token_to_idx[tok] = idx
                self.idx_to_token[idx] = tok
        else:
            # 5 specials + the 256 possible UTF-8 byte values, so token id = 5 + byte value. Any
            # Unicode character is representable as a byte sequence, so this inventory never
            # changes and no vocabulary file is involved. [UNK] is unreachable.
            self._add_special_tokens()
            self._register_chars(chr(b) for b in range(256))

        self.tokens_list = [self.idx_to_token[i] for i in range(len(self.idx_to_token))]

        self.beam_width = 10
        self.blank_penalty = 0.0
        self.length_penalty = 0.0
        self.nbest = 1
        self.beam_threshold = 50.0
        self.lm_weight = lm_weight
        self.word_score = 0.0
        self.unk_score = float("-inf")
        self.log_add = True

        self._decoder_cache = {}

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def _get_decoder(self, beam_width: int, blank_penalty: float, nbest: int):
        """The LM settings are part of the cache key so that changing lm_weight between calls
        rebuilds the decoder instead of silently reusing the previous weight."""
        key = (beam_width, blank_penalty, nbest, self.log_add, self.lm_weight,
               str(self.kenlm_path), str(self.lm_dict_path))
        if key not in self._decoder_cache:
            if self.verbose:
                print(f"Building CTC decoder: beam={beam_width}, nbest={nbest}, lm_weight={self.lm_weight}")
            self._decoder_cache[key] = self._build_decoder(beam_width, blank_penalty, nbest, self.log_add)
        return self._decoder_cache[key]

    def _build_decoder(self, beam_width: int, blank_penalty: float, nbest: int, log_add: bool):
        """lm_dict_path must accompany kenlm_path: without it torchaudio derives the LM vocabulary
        from tokens_list, but ' ' and '\\n' cannot be words in a whitespace-delimited ARPA file
        and would resolve to <unk> on every lookup. build_kenlm.py emits an index-parallel dict
        that aliases exactly those tokens."""
        from torchaudio.models.decoder import ctc_decoder
        if self.kenlm_path and not self.lm_dict_path:
            raise ValueError(
                "kenlm_path is set but lm_dict_path is not. The LM would be queried with a "
                "token inventory it was not trained on (' ' and '\\n' resolving to <unk>). "
                "Pass the lm_dict emitted alongside the LM by build_kenlm.py."
            )
        return ctc_decoder(
            lexicon=None,
            tokens=self.tokens_list,
            lm=str(self.kenlm_path) if self.kenlm_path else None,
            lm_dict=str(self.lm_dict_path) if self.lm_dict_path else None,
            nbest=nbest,
            beam_size=beam_width,
            beam_size_token=None,
            beam_threshold=self.beam_threshold,
            lm_weight=self.lm_weight,
            word_score=self.word_score,
            unk_score=self.unk_score,
            sil_score=blank_penalty,
            log_add=log_add,
            blank_token=self.blank_token,
            sil_token=self.blank_token,
            unk_word=self.unk_token
        )

    def _add_special_tokens(self):
        self.token_to_idx[self.blank_token] = self.blank_token_id
        self.idx_to_token[self.blank_token_id] = self.blank_token
        self.token_to_idx[self.unk_token] = self.unk_token_id
        self.idx_to_token[self.unk_token_id] = self.unk_token
        self.token_to_idx[self.pad_token] = self.pad_token_id
        self.idx_to_token[self.pad_token_id] = self.pad_token
        self.token_to_idx[self.bos_token] = self.bos_token_id
        self.idx_to_token[self.bos_token_id] = self.bos_token
        self.token_to_idx[self.eos_token] = self.eos_token_id
        self.idx_to_token[self.eos_token_id] = self.eos_token

    def _register_chars(self, chars):
        next_idx = len(self.token_to_idx)
        for char in chars:
            if char not in self.token_to_idx:
                self.token_to_idx[char] = next_idx
                self.idx_to_token[next_idx] = char
                next_idx += 1

    def encode(self, text: str, add_special_tokens=False) -> list[int]:
        if self.normalize_unicode:
            text = canonicalize(text)
        ids = [self.token_to_idx.get(chr(b), self.unk_token_id) for b in text.encode('utf-8')]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def _tokens_to_text(self, tokens: list[str]) -> str:
        """Byte runs decode with errors='replace', so an invalid sequence the model emits surfaces
        as U+FFFD and counts as a CER error instead of crashing."""
        parts = []
        run = bytearray()
        for tok in tokens:
            if len(tok) == 1:
                run.append(ord(tok))
            else:
                if run:
                    parts.append(run.decode('utf-8', errors='replace'))
                    run = bytearray()
                parts.append(tok)
        if run:
            parts.append(run.decode('utf-8', errors='replace'))
        return ''.join(parts)

    def decode(self, ids: torch.Tensor | list[int], skip_special_tokens=True) -> str:
        id_list = ids.cpu().tolist() if isinstance(ids, torch.Tensor) else ids
        tokens = [self.idx_to_token.get(id_int, self.unk_token) for id_int in id_list]

        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        return self._tokens_to_text(tokens)

    def batch_decode(self, batch_ids: list[torch.Tensor | list[int]], skip_special_tokens=True) -> list[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    def ctc_decode(self, ids: torch.Tensor | list[int], skip_special_tokens=True) -> str:
        id_list = ids.cpu().tolist() if isinstance(ids, torch.Tensor) else ids

        collapsed = [k for k, _ in itertools.groupby(id_list) if k != self.blank_token_id]

        tokens = [self.idx_to_token.get(id_int, self.unk_token) for id_int in collapsed]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        return self._tokens_to_text(tokens)

    def batch_ctc_decode(self, batch_ids: list[torch.Tensor | list[int]], skip_special_tokens=True) -> list[str]:
        return [self.ctc_decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    def batch_decode_beam_search(
        self,
        log_probs: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
        beam_width: int | None = None,
        blank_penalty: float | None = None,
        length_penalty: float | None = None,
    ) -> list[str]:
        bw = beam_width if beam_width is not None else self.beam_width
        bp = blank_penalty if blank_penalty is not None else self.blank_penalty
        lp = length_penalty if length_penalty is not None else self.length_penalty

        required_nbest = self.nbest
        if lp != 0.0 and self.nbest == 1:
            required_nbest = min(bw, 10)

        decoder = self._get_decoder(bw, bp, required_nbest)

        emissions = log_probs.permute(1, 0, 2).float().contiguous().cpu()
        lengths = input_lengths.cpu() if input_lengths is not None else None

        batch_hypos = decoder(emissions, lengths)
        batch_texts = []

        for hypos in batch_hypos:
            if not hypos:
                batch_texts.append("")
                continue

            if lp != 0.0 and len(hypos) > 1:
                adjusted_hypos = []
                for hypo in hypos:
                    tokens = [self.tokens_list[idx] for idx in hypo.tokens]
                    text = self._tokens_to_text([t for t in tokens if t not in self.special_tokens])

                    length = max(len(text), 1)
                    adjusted_score = hypo.score / (length**lp)
                    adjusted_hypos.append((adjusted_score, text))

                adjusted_hypos.sort(key=lambda x: x[0], reverse=True)
                batch_texts.append(adjusted_hypos[0][1])
            else:
                top_hypo = hypos[0]
                tokens = [self.tokens_list[idx] for idx in top_hypo.tokens]
                text = self._tokens_to_text([t for t in tokens if t not in self.special_tokens])
                batch_texts.append(text)

        return batch_texts

    def batch_beam_search_nbest(
        self,
        log_probs: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
        beam_width: int | None = None,
        nbest: int = 8,
    ) -> list[list[tuple[str, float]]]:
        """Returns (text, ctc_score) tuples per sample, best first."""
        bw = beam_width if beam_width is not None else self.beam_width
        decoder = self._get_decoder(bw, self.blank_penalty, nbest)

        emissions = log_probs.permute(1, 0, 2).float().contiguous().cpu()
        lengths = input_lengths.cpu() if input_lengths is not None else None

        batch_hypos = decoder(emissions, lengths)
        results = []
        for hypos in batch_hypos:
            sample = []
            for hypo in hypos:
                tokens = [self.tokens_list[idx] for idx in hypo.tokens]
                text = self._tokens_to_text([t for t in tokens if t not in self.special_tokens])
                sample.append((text, float(hypo.score)))
            results.append(sample)
        return results
