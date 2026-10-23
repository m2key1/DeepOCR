import json
from collections import defaultdict
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from jiwer import cer as jiwer_cer, wer as jiwer_wer
from PIL import Image

from utils.dataset import derive_dataset_label
from utils.model import DeepOCR
from utils.tokenizer import DeepTokenizer, check_kenlm_artifacts

CHECKPOINT_PATH = Path('runs/latest_run/best_model.pt') # train.py keeps runs/latest_run pointing at the newest run
VALID_DIR = Path('data/valid')
MANIFEST_PATH = Path('data/split_manifest.json')
DATASET_SOURCES = []
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
MAX_SAMPLES = 200
DECODER_MAX_LEN = 256
RUN_GREEDY = True
RUN_BEAM = True
RUN_DECODER = False # adds almost nothing once the KenLM is fused into the beam
RUN_RESCORED = False
BEAM_WIDTH = 10
RESCORE_BEAM_WIDTH = 4
RESCORE_NBEST = 8
RESCORE_ALPHA = 0.3
RESCORE_BETA = 0.0
# One byte-level KenLM over all training domains, the same one train.py fuses. It helps every prose
# domain and is neutral on structured data. Build it with build_kenlm.py.
KENLM_PATH = Path('data/lm/union_bytes_v3.bin')
LM_DICT_PATH = Path('data/lm/union_bytes_v3.dict') # index-parallel to the tokenizer; required alongside KENLM_PATH
LM_WEIGHT = 1.0 # swept for the byte LM on the base model; drop to ~0.5 for mostly-numeric/structured text
NUM_EXAMPLES_TO_PRINT = 20
# Which transcription convention to ask for, when the checkpoint was trained with conditioning
# (model.convention_names lists what it understands). None means the normalized convention.
CONVENTION: dict[str, float] | None = None
MIN_FREE_GPU_BYTES = 4 * 1024 ** 3


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes > MIN_FREE_GPU_BYTES:
            return torch.device('cuda')
    return torch.device('cpu')


DEVICE = pick_device()


def load_model(checkpoint_path: str, device: torch.device) -> tuple[DeepOCR, DeepTokenizer]:
    # The tokenizer inventory lives in the checkpoint, so the LM can only be checked against it
    # after loading; load once without, verify, and attach if the pair is index-parallel.
    model, tokenizer = DeepOCR.load(checkpoint_path, device=device)
    kenlm_path, lm_dict_path = check_kenlm_artifacts(KENLM_PATH, LM_DICT_PATH, tokenizer.tokens_list)
    if kenlm_path:
        model, tokenizer = DeepOCR.load(
            checkpoint_path,
            device=device,
            kenlm_path=kenlm_path,
            lm_dict_path=lm_dict_path,
            lm_weight=LM_WEIGHT,
        )
    model.eval()
    return model, tokenizer


_NORMALIZE = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def convention_tensor(model: DeepOCR, batch_size: int, device: torch.device) -> torch.Tensor | None:
    """Repeats the requested convention flags across the batch. Unknown names are rejected rather
    than silently ignored, since a typo would quietly give the default convention instead."""
    if not model.convention_names:
        return None
    unknown = set(CONVENTION or ()) - set(model.convention_names)
    if unknown:
        raise ValueError(f"CONVENTION has names this checkpoint does not know: {sorted(unknown)}; "
                         f"it understands {model.convention_names}")
    row = [float((CONVENTION or {}).get(name, 0.0)) for name in model.convention_names]
    return torch.tensor([row] * batch_size, device=device)


def preprocess(image: Image.Image, model: DeepOCR) -> torch.Tensor:
    image = image.convert('RGB')
    width, height = image.size
    new_width = max(1, int(width * (model.target_height / height)))
    if model.max_width and new_width > model.max_width:
        new_width = model.max_width
    image = image.resize((new_width, model.target_height), Image.Resampling.LANCZOS)
    return _NORMALIZE(image=np.array(image))['image']


@torch.no_grad()
def transcribe(model: DeepOCR, tokenizer: DeepTokenizer, images: list[Image.Image],
               device: torch.device, use_decoder: bool) -> list[str]:
    predictions = []
    for image in images:
        tensor = preprocess(image, model).unsqueeze(0).to(device)
        memory, memory_key_padding_mask, *_ = model._encode(
            tensor, conventions=convention_tensor(model, 1, device))
        if use_decoder:
            generated = model.decode_autoregressive(memory, memory_key_padding_mask, max_len=DECODER_MAX_LEN)
            predictions.append(model._decoder_ids_to_text(generated)[0])
        else:
            logits = model.output_layer(memory)
            pred_ids = logits.argmax(dim=-1).squeeze(1)
            predictions.append(tokenizer.ctc_decode(pred_ids))
    return predictions


@torch.no_grad()
def transcribe_beam(model: DeepOCR, tokenizer: DeepTokenizer, images: list[Image.Image],
                    device: torch.device) -> list[str]:
    """CTC beam search with the KenLM that load_model attached, or plain beam search when none
    was found."""
    predictions = []
    for image in images:
        tensor = preprocess(image, model).unsqueeze(0).to(device)
        memory, _, *_ = model._encode(tensor, conventions=convention_tensor(model, 1, device))
        log_probs = F.log_softmax(model.output_layer(memory), dim=-1)
        predictions.append(tokenizer.batch_decode_beam_search(log_probs, beam_width=BEAM_WIDTH)[0])
    return predictions


@torch.no_grad()
def transcribe_rescored(model: DeepOCR, tokenizer: DeepTokenizer, images: list[Image.Image],
                        device: torch.device) -> list[str]:
    predictions = []
    for image in images:
        tensor = preprocess(image, model).unsqueeze(0).to(device)
        memory, memory_key_padding_mask, *_ = model._encode(
            tensor, conventions=convention_tensor(model, 1, device))
        log_probs = F.log_softmax(model.output_layer(memory), dim=-1)
        input_lengths = torch.tensor([log_probs.size(0)], dtype=torch.long)
        nbest_lists = tokenizer.batch_beam_search_nbest(
            log_probs, input_lengths, beam_width=RESCORE_BEAM_WIDTH, nbest=RESCORE_NBEST
        )
        predictions.append(model.rescore_hypotheses(
            memory, memory_key_padding_mask, nbest_lists, alpha=RESCORE_ALPHA, beta=RESCORE_BETA
        )[0])
    return predictions


def read_ground_truth(image_path: Path) -> str | None:
    for suffix in ('.gt.txt', '.txt'):
        candidate = image_path.with_name(image_path.stem + suffix)
        if candidate.exists():
            return candidate.read_text(encoding='utf-8').strip()
    return None


def collect_samples() -> list[tuple[Path, str, str]]:
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    valid_sources = manifest.get('valid', {})

    samples = []
    for image_path in sorted(VALID_DIR.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        ground_truth = read_ground_truth(image_path)
        if not ground_truth:
            continue
        source_path = valid_sources.get(image_path.stem, str(image_path))
        label = derive_dataset_label(source_path, DATASET_SOURCES)
        samples.append((image_path, ground_truth, label))
        if len(samples) >= MAX_SAMPLES:
            break
    return samples


def summarize(references: list[str], hypotheses: list[str]) -> tuple[float, float]:
    if not references:
        return float('nan'), float('nan')
    return jiwer_cer(references, hypotheses), jiwer_wer(references, hypotheses)


def main():
    samples = collect_samples()
    print(f"Device: {DEVICE} | checkpoint: {CHECKPOINT_PATH}")
    print(f"Collected {len(samples)} validation samples from {VALID_DIR}\n")
    if not samples:
        return

    images = [Image.open(p).convert('RGB') for p, _, _ in samples]
    references = [gt for _, gt, _ in samples]
    labels = [label for _, _, label in samples]

    model, tokenizer = load_model(str(CHECKPOINT_PATH), DEVICE)

    greedy = transcribe(model, tokenizer, images, DEVICE, use_decoder=False) if RUN_GREEDY else None
    beam = transcribe_beam(model, tokenizer, images, DEVICE) if RUN_BEAM else None
    decoder = transcribe(model, tokenizer, images, DEVICE, use_decoder=True) if RUN_DECODER else None
    rescored = transcribe_rescored(model, tokenizer, images, DEVICE) if RUN_RESCORED and model.use_decoder else None

    print("=" * 100)
    for i in range(min(NUM_EXAMPLES_TO_PRINT, len(samples))):
        print(f"[{labels[i]}]")
        print(f"  gt      : {references[i]}")
        if greedy is not None:
            print(f"  greedy  : {greedy[i]}")
        if beam is not None:
            print(f"  beam+LM : {beam[i]}")
        if decoder is not None:
            print(f"  decoder : {decoder[i]}")
        if rescored is not None:
            print(f"  rescored: {rescored[i]}")
        print("-" * 100)

    print("\n=== overall ===")
    if greedy is not None:
        cer, wer = summarize(references, greedy)
        print(f"greedy  : CER {cer:.4f}  WER {wer:.4f}")
    if beam is not None:
        cer, wer = summarize(references, beam)
        print(f"beam+LM : CER {cer:.4f}  WER {wer:.4f}")
    if decoder is not None:
        cer, wer = summarize(references, decoder)
        print(f"decoder : CER {cer:.4f}  WER {wer:.4f}")
    if rescored is not None:
        cer, wer = summarize(references, rescored)
        print(f"rescored: CER {cer:.4f}  WER {wer:.4f}")

    print("\n=== per-dataset (worst CER first) ===")
    by_label = defaultdict(lambda: {'ref': [], 'greedy': [], 'beam': [], 'decoder': [], 'rescored': []})
    for i, label in enumerate(labels):
        by_label[label]['ref'].append(references[i])
        if greedy is not None:
            by_label[label]['greedy'].append(greedy[i])
        if beam is not None:
            by_label[label]['beam'].append(beam[i])
        if decoder is not None:
            by_label[label]['decoder'].append(decoder[i])
        if rescored is not None:
            by_label[label]['rescored'].append(rescored[i])

    rows = []
    for label, bucket in by_label.items():
        g_cer = summarize(bucket['ref'], bucket['greedy'])[0] if greedy is not None else float('nan')
        b_cer = summarize(bucket['ref'], bucket['beam'])[0] if beam is not None else float('nan')
        d_cer = summarize(bucket['ref'], bucket['decoder'])[0] if decoder is not None else float('nan')
        r_cer = summarize(bucket['ref'], bucket['rescored'])[0] if rescored is not None else float('nan')
        rows.append((label, g_cer, b_cer, d_cer, r_cer, len(bucket['ref'])))

    # x == x is False for NaN, so datasets without a score sort last
    rank_index = 2 if beam is not None else 1
    rows.sort(key=lambda r: r[rank_index] if r[rank_index] == r[rank_index] else -1.0, reverse=True)
    header = "{:<44}{:>10}{:>10}{:>10}{:>10}{:>7}".format('dataset', 'greedyCER', 'beamCER', 'decCER', 'rescCER', 'n')
    print(header)
    print("-" * len(header))
    for label, g_cer, b_cer, d_cer, r_cer, n in rows:
        print("{:<44}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>7d}".format(label, g_cer, b_cer, d_cer, r_cer, n))


if __name__ == '__main__':
    main()
