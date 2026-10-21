import sys
from pathlib import Path
from uuid import uuid4

from datasets import load_dataset

DATASET_NAME = 'rjadr/NotarialDeeds_HTR'
SPLIT = 'train'
IMAGE_COLUMN = 'image'
TEXT_COLUMN = 'text'
OUTPUT_DIR = Path('data_sources/NotarialDeeds_HTR')


def main():
    dataset = load_dataset(DATASET_NAME)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for example in dataset[SPLIT]:
        image = example[IMAGE_COLUMN]
        text = example[TEXT_COLUMN]
        if image is None or text is None or image.size == (0, 0) or not text.strip():
            continue
        image_path = OUTPUT_DIR / f"{uuid4()}.png"
        image.save(image_path)
        image_path.with_name(f"{image_path.stem}.gt.txt").write_text(text, encoding='utf-8')
        written += 1
    print(f"{DATASET_NAME}[{SPLIT}]: wrote {written} line pairs to {OUTPUT_DIR}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
