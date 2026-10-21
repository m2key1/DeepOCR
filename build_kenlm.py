r"""Trains a character- or byte-level KenLM for CTC beam-search shallow fusion.

flashlight queries the LM once per token, lexicon-free, and maps CTC output index -> lm_dict
index -> KenLM vocab id positionally, so the LM vocabulary must be index-parallel to the
tokenizer. ARPA files are whitespace-delimited, so tokens that are whitespace (' ', '\n', '\xa0')
are renamed to <space>/<nl>/<nbsp> in both the training text and the lm_dict. Without that, every
space would score as <unk>.

Writes <out>.bin (or <out>.arpa when build_binary is missing; flashlight reads ARPA directly) and
<out>.dict; pass both to DeepTokenizer as kenlm_path/lm_dict_path.

data/train is flat, so source filtering with --include/--exclude needs --manifest.

Usage:
    python build_kenlm.py --input data/train --out data/lm/catmus --order 6 \
        --manifest data/split_manifest.json --include catmus_medieval \
        --checkpoint runs/<run>/best_model.pt

    # where lmplz is unavailable: emit the corpus, build the ARPA elsewhere
    python build_kenlm.py --input data/train --out /tmp/catmus --corpus-only ...
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

from utils.tokenizer import DeepTokenizer, canonicalize

# Whitespace tokens cannot be ARPA words; these stand in for them in the corpus and the lm_dict.
WHITESPACE_ALIASES = {' ': '<space>', '\n': '<nl>', '\t': '<tab>', '\xa0': '<nbsp>'}
# KenLM owns these; a token colliding with one would corrupt the model's own boundary handling.
KENLM_RESERVED = {'<unk>', '<s>', '</s>'}
# The KenLM bundled in the flashlight-text wheel is compiled with KENLM_MAX_ORDER=6. A higher-order
# ARPA builds without complaint and only fails when the decoder loads it.
KENLM_MAX_ORDER = 6


def resolve_tokens(args) -> list[str]:
    """A checkpoint wins, since a character-vocabulary checkpoint carries its own inventory."""
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        tokens = ck.get('tokenizer_tokens')
        if not tokens:
            sys.exit(f"{args.checkpoint} has no 'tokenizer_tokens'.")
        print(f"Token inventory: {len(tokens)} tokens from checkpoint {args.checkpoint}")
        return list(tokens)

    tok = DeepTokenizer()
    print(f"Token inventory: {len(tok.tokens_list)} tokens (byte vocabulary)")
    return list(tok.tokens_list)


def is_byte_inventory(tokens: list[str]) -> bool:
    return len(tokens) == 261 and tokens[5:] == [chr(b) for b in range(256)]


def build_alias_map(tokens: list[str]) -> dict[str, str]:
    alias = {}
    taken = set(tokens)
    for tok in tokens:
        # not isprintable() covers the control and format characters the byte vocabulary
        # contains (C0/C1 bytes, DEL, soft hyphen); verbatim they would corrupt the ARPA file.
        needs_alias = any(c.isspace() for c in tok) or not tok.isprintable() or tok in KENLM_RESERVED
        if not needs_alias:
            continue
        candidate = WHITESPACE_ALIASES.get(tok) or f"<tok{tokens.index(tok)}>"
        if candidate in taken:
            candidate = f"<tok{tokens.index(tok)}>"
        alias[tok] = candidate
        taken.add(candidate)
    return alias


def write_lm_dict(tokens: list[str], alias: dict[str, str], path: Path):
    """Writes the index-parallel LM dictionary: line N is the LM's name for token id N."""
    lines = [alias.get(tok, tok) for tok in tokens]
    for i, line in enumerate(lines):
        if not line or any(c.isspace() for c in line):
            raise RuntimeError(f"lm_dict entry {i} is not whitespace-free: {line!r}")
    if len(set(lines)) != len(lines):
        dupes = [t for t, n in Counter(lines).items() if n > 1]
        raise RuntimeError(f"lm_dict entries must be unique; duplicated: {dupes}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"Wrote lm_dict: {path} ({len(lines)} entries, {len(alias)} aliased)")


def collect_corpus(input_dirs: list[Path], include: list[str], exclude: list[str],
                   manifest: Path | None, manifest_split: str) -> list[Path]:
    """With a manifest, --include/--exclude match against the original data_sources/ path,
    since data/train is flat."""
    # The split keeps each sample's original suffix, .gt.txt or .txt. One text file per sample, so
    # *.txt never double-counts.
    files = []
    for d in input_dirs:
        if not d.exists():
            sys.exit(f"Input dir not found: {d}")
        files.extend(sorted(d.rglob('*.txt')))

    if not (include or exclude):
        return files

    if manifest:
        mapping = json.loads(manifest.read_text())[manifest_split]
        def key_of(f: Path) -> str:
            stem = f.name
            for suffix in ('.gt.txt', '.corrected.txt', '.txt'):
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break
            return mapping.get(stem, '')
        unmapped = sum(1 for f in files if not key_of(f))
        if unmapped:
            print(f"WARNING: {unmapped}/{len(files)} files absent from manifest['{manifest_split}']; "
                  f"they cannot be source-filtered and are dropped.")
    else:
        def key_of(f: Path) -> str:
            return str(f)

    if include:
        files = [f for f in files if any(pat in key_of(f) for pat in include)]
    if exclude:
        files = [f for f in files if not any(pat in key_of(f) for pat in exclude)]
    return files


def write_corpus(files: list[Path], tokens: list[str], alias: dict[str, str],
                 unk_token: str, path: Path, use_bytes: bool = False) -> dict:
    """Renders each line as space-separated tokens, canonicalised like DeepTokenizer.encode.
    Out-of-vocabulary characters become UNK rather than being dropped, so the n-gram context
    stays aligned."""
    vocab = set(tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    with path.open('w', encoding='utf-8') as out:
        for f in files:
            try:
                text = f.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                stats['unreadable'] += 1
                continue
            text = canonicalize(text).strip('\n')
            if not text.strip():
                stats['empty'] += 1
                continue
            units = [chr(b) for b in text.encode('utf-8')] if use_bytes else text
            rendered = []
            for ch in units:
                if ch in vocab:
                    rendered.append(alias.get(ch, ch))
                else:
                    rendered.append(alias.get(unk_token, unk_token))
                    stats['oov_chars'] += 1
            out.write(' '.join(rendered) + '\n')
            stats['lines'] += 1
            stats['chars'] += len(rendered)
    return stats


def write_raw_text(files: list[Path], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open('w', encoding='utf-8') as out:
        for f in files:
            try:
                text = f.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                continue
            text = ' '.join(canonicalize(text).split())
            if text:
                out.write(text + '\n')
                n += 1
    return n


def find_kenlm_bin(name: str, kenlm_bin: str | None, required: bool = True) -> str | None:
    if kenlm_bin:
        cand = Path(kenlm_bin) / name
        if cand.exists():
            return str(cand)
        if not required:
            return None
        sys.exit(f"{name} not found in --kenlm-bin {kenlm_bin}")
    found = shutil.which(name)
    if not found:
        if not required:
            return None
        sys.exit(
            f"{name} not on PATH. KenLM's trainer is a C++ binary and is NOT included in the\n"
            f"`kenlm` pip package (that ships the query API only). Build it:\n"
            f"    git clone https://github.com/kpu/kenlm && cd kenlm\n"
            f"    cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8 --target lmplz build_binary\n"
            f"then re-run with --kenlm-bin kenlm/build/bin\n"
            f"\n"
            f"On Boost >= 1.69 that cmake fails with 'Could not find boost_system': Boost.System is\n"
            f"header-only now, and unit_test_framework is only needed for KenLM's own tests. Drop\n"
            f"both from the COMPONENTS list in kenlm/CMakeLists.txt and it configures.\n"
            f"Without a compiler toolchain, use --corpus-only here and build the ARPA on a machine\n"
            f"that has lmplz; the corpus is plain text."
        )
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', nargs='+', type=Path, default=[Path('data/train')],
                   help="Dirs to rglob for *.txt. Default data/train; data/valid or data_sources "
                        "would leak validation text into the LM.")
    p.add_argument('--out', type=Path, required=True, help="Output prefix; writes <out>.bin and <out>.dict")
    p.add_argument('--order', type=int, default=6,
                   help="n-gram order, default 6. The flashlight-text wheel's KenLM is compiled with "
                        "KENLM_MAX_ORDER=6; a higher order builds but fails at decode time.")
    p.add_argument('--checkpoint', type=Path, default=None,
                   help="Checkpoint to take the token inventory from (preferred)")
    p.add_argument('--include', nargs='*', default=[], help="Only files whose source path contains one of these")
    p.add_argument('--exclude', nargs='*', default=[], help="Skip files whose source path contains one of these")
    p.add_argument('--manifest', type=Path, default=None,
                   help="split_manifest.json, mapping sample id -> data_sources/ path. Needed for "
                        "--include/--exclude on a flat input dir such as data/train.")
    p.add_argument('--manifest-split', default='train', help="Manifest split key to read (default: train)")
    p.add_argument('--corpus-only', action='store_true',
                   help="Write the corpus and lm_dict, then stop. For machines without lmplz.")
    p.add_argument('--dump-raw-text', type=Path, default=None,
                   help="Also write the selected lines as plain canonicalised text, one per line, "
                        "for pretrain_decoder.py --text-file.")
    p.add_argument('--prune', nargs='*', type=int, default=None, help="lmplz --prune (rarely needed for char LMs)")
    p.add_argument('--memory', default='40%', help="lmplz -S memory budget")
    p.add_argument('--kenlm-bin', default=None, help="Dir containing lmplz/build_binary")
    p.add_argument('--keep-intermediate', action='store_true', help="Keep the corpus and .arpa files")
    args = p.parse_args()

    if args.order > KENLM_MAX_ORDER:
        print(f"WARNING: --order {args.order} exceeds the KENLM_MAX_ORDER={KENLM_MAX_ORDER} that the "
              f"stock flashlight-text wheel is compiled with. lmplz will build it, but loading it for "
              f"decoding will raise FormatLoadException unless flashlight-text is rebuilt from source.")

    tokens = resolve_tokens(args)
    unk_token = tokens[1]  # DeepTokenizer pins [UNK] to id 1

    use_bytes = is_byte_inventory(tokens)
    if use_bytes:
        print("Byte vocabulary detected; corpus will be rendered as UTF-8 byte tokens.")

    alias = build_alias_map(tokens)
    print(f"Aliased {len(alias)} unrepresentable tokens: "
          + ', '.join(f'{t!r}->{a}' for t, a in alias.items()))

    dict_path = Path(f"{args.out}.dict")
    write_lm_dict(tokens, alias, dict_path)

    files = collect_corpus(args.input, args.include, args.exclude, args.manifest, args.manifest_split)
    if not files:
        sys.exit(f"No *.txt found under {[str(d) for d in args.input]} with the given filters.")
    print(f"Corpus: {len(files)} files from {[str(d) for d in args.input]}")

    if args.dump_raw_text:
        n = write_raw_text(files, args.dump_raw_text)
        print(f"Wrote raw text: {args.dump_raw_text} ({n} lines)")

    corpus_path = Path(f"{args.out}.corpus.txt")
    stats = write_corpus(files, tokens, alias, unk_token, corpus_path, use_bytes=use_bytes)
    print(f"Corpus: {stats['lines']} lines, {stats['chars']} char tokens, "
          f"{stats['oov_chars']} OOV->UNK, {stats['empty']} empty skipped")
    if not stats['lines']:
        sys.exit("Corpus is empty; nothing to train on.")
    if stats['chars'] < 100_000:
        print(f"WARNING: only {stats['chars']} char tokens. A {args.order}-gram LM will be "
              f"badly undertrained; consider a lower --order or more data.")

    if args.corpus_only:
        print(f"\n--corpus-only: stopping here.\n  corpus:  {corpus_path}\n  lm_dict: {dict_path}\n"
              f"Build the ARPA elsewhere with:\n"
              f"    lmplz -o {args.order} --text <corpus> --arpa <out>.arpa [--discount_fallback]")
        return

    lmplz = find_kenlm_bin('lmplz', args.kenlm_bin)
    build_binary = find_kenlm_bin('build_binary', args.kenlm_bin, required=False)

    arpa_path = Path(f"{args.out}.arpa")
    cmd = [lmplz, '-o', str(args.order), '-S', args.memory, '--text', str(corpus_path),
           '--arpa', str(arpa_path)]
    if args.prune:
        cmd += ['--prune'] + [str(x) for x in args.prune]

    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        # Kneser-Ney discount estimation needs well-populated count-of-counts; a small or
        # highly repetitive char corpus routinely degenerates, and this is the documented escape.
        print("\nlmplz failed; retrying with --discount_fallback (common for char corpora)")
        res = subprocess.run(cmd + ['--discount_fallback'])
        if res.returncode != 0:
            sys.exit("lmplz failed even with --discount_fallback.")

    if build_binary:
        lm_path = Path(f"{args.out}.bin")
        print(f"\n$ {build_binary} trie {arpa_path} {lm_path}")
        if subprocess.run([build_binary, 'trie', str(arpa_path), str(lm_path)]).returncode != 0:
            sys.exit("build_binary failed.")
        if not args.keep_intermediate:
            arpa_path.unlink(missing_ok=True)
    else:
        lm_path = arpa_path
        print("\nbuild_binary not found; keeping the .arpa and using it directly. "
              "flashlight's KenLM reads ARPA, so this only costs load time, not accuracy.")

    if not args.keep_intermediate:
        corpus_path.unlink(missing_ok=True)

    print(f"\nDone.\n  LM:      {lm_path}\n  lm_dict: {dict_path}\n\nUse with:\n"
          f"    KENLM_PATH = Path('{lm_path}')\n    LM_DICT_PATH = Path('{dict_path}')")


if __name__ == '__main__':
    main()
