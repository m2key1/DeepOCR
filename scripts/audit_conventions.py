import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import DATASET_SOURCES, DROP_DATASETS, SPLIT_MANIFEST, TRAIN_DATA_DIR
from utils.dataset import derive_dataset_label

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
TEXT_EXTENSIONS = ('.txt', '.gt.txt', '.corrected.txt')
MIN_LINES = 200
# Diplomatic and normalized corpora fall far either side of this, so the cut sits in the gap.
MARKED_RATIO = 0.10
OUT_PATH = Path('data/conventions.json')

# Each flag pairs the diplomatic form with its normalized counterpart. A corpus
# gets the flag when it writes the diplomatic form on more than MARKED_RATIO of its occurrences.
FLAGS = {
    'long_s': ('ſ', 's'),
    'oblique_hyphen': ('⸗', '-'),
    'typographic_punctuation': ('’‘”–', "'\"-"),
}


def iter_labelled_texts():
    """Reads the prepared training split rather than data_sources, so every corpus is measured on
    the text it trains on, whichever format (loose text, PAGE or ALTO XML) it shipped in."""
    stem_to_source = {}
    if SPLIT_MANIFEST.exists():
        manifest = json.loads(SPLIT_MANIFEST.read_text(encoding='utf-8'))
        stem_to_source = manifest.get('train', {})
    dropped = set(DROP_DATASETS)
    names = {entry.name for entry in os.scandir(TRAIN_DATA_DIR)}
    for name in names:
        stem, extension = os.path.splitext(name)
        if extension.lower() not in IMAGE_EXTENSIONS:
            continue
        text_name = next((stem + ext for ext in TEXT_EXTENSIONS if stem + ext in names), None)
        if text_name is None:
            continue
        source = stem_to_source.get(stem, str(TRAIN_DATA_DIR / name))
        if dropped & set(source.replace('\\', '/').split('/')):
            continue
        label = derive_dataset_label(source, DATASET_SOURCES)
        yield label, (TRAIN_DATA_DIR / text_name).read_text(encoding='utf-8', errors='ignore')


def main():
    counts = collections.defaultdict(collections.Counter)
    for label, text in iter_labelled_texts():
        bucket = counts[label]
        bucket['lines'] += 1
        for flag, (marked, plain) in FLAGS.items():
            bucket[flag] += sum(text.count(c) for c in marked)
            bucket[f'{flag}_plain'] += sum(text.count(c) for c in plain)

    table = {}
    print(f"\n{'dataset label':<42} {'lines':>8}  " + '  '.join(f'{f:>24}' for f in FLAGS))
    for label, bucket in sorted(counts.items(), key=lambda kv: -kv[1]['lines']):
        if bucket['lines'] < MIN_LINES:
            continue
        cells, row = [], []
        for flag in FLAGS:
            marked, plain = bucket[flag], bucket[f'{flag}_plain']
            ratio = marked / max(1, marked + plain)
            row.append(1.0 if ratio > MARKED_RATIO else 0.0)
            cells.append(f"{marked:>10,} {ratio:>6.1%}")
        print(f"{label:<42} {bucket['lines']:>8,}  " + '  '.join(cells))
        if any(row):
            table[label] = row

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({'names': list(FLAGS), 'by_dataset': table}, indent=2,
                                   ensure_ascii=False) + '\n')
    print(f"\n{len(table)} of {len(counts)} labels are non-default; wrote {OUT_PATH}")
    print("train.py loads it as CONVENTION_BY_DATASET.")


if __name__ == '__main__':
    main()
