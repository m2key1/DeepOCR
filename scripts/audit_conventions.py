import collections
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import DATASET_SOURCES
from utils.dataset import derive_dataset_label

SOURCES_ROOT = Path('data_sources')
MIN_LINES = 200
# Diplomatic and normalized corpora fall far either side of this, so the cut sits in the gap.
MARKED_RATIO = 0.10
SAMPLE_PER_SOURCE = 40000
OUT_PATH = Path('data/conventions.json')

# Each flag pairs the diplomatic form with its normalized counterpart. A corpus
# gets the flag when it writes the diplomatic form on more than MARKED_RATIO of its occurrences.
FLAGS = {
    'long_s': ('ſ', 's'),
    'oblique_hyphen': ('⸗', '-'),
    'typographic_punctuation': ('’‘”–', "'\"-"),
}


def iter_texts(source: Path):
    """Ground truth as the dataset sees it: loose transcriptions plus PAGE XML text lines, which
    is where the handwriting corpora keep theirs."""
    seen = 0
    for path in source.rglob('*'):
        if seen >= SAMPLE_PER_SOURCE:
            return
        if path.suffix.lower() == '.txt':
            try:
                yield path, path.read_text(encoding='utf-8', errors='ignore')
                seen += 1
            except OSError:
                continue
        elif path.suffix.lower() == '.xml':
            try:
                root = ET.fromstring(path.read_text(encoding='utf-8', errors='ignore'))
            except (OSError, ET.ParseError):
                continue
            for node in root.iter():
                if node.tag.rsplit('}', 1)[-1] == 'Unicode' and node.text:
                    yield path, node.text
                    seen += 1


def main():
    counts = collections.defaultdict(collections.Counter)
    for entry in DATASET_SOURCES:
        source = Path(entry)
        if not source.exists():
            print(f"  skipped, not on this machine: {entry}")
            continue
        for path, text in iter_texts(source):
            label = derive_dataset_label(str(path), DATASET_SOURCES)
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
