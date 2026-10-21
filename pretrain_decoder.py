import argparse
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import PositionalEncoding
from utils.tokenizer import DeepTokenizer

D_MODEL = 512
DECODER_LAYERS = 4
DECODER_HEADS = 8
DECODER_FFN_DIM = 2048
DECODER_DROPOUT = 0.1
MAX_SEQ_LEN = 5000

TEXT_DATA_DIRS = [Path('data_sources')]
EXTERNAL_TEXT_FILES = []
EXCLUDE_DATA = ['kurrent', 'READ', 'kompass', 'compass']
INCLUDE_DATA = []
VALID_SPLIT = 0.02
SPLIT_SEED = 42

BATCH_SIZE = 512 # (512 ~ 22GB VRAM)
EPOCHS = 30
LR = 3e-4
WARMUP_STEPS = 500
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 5.0
MIN_LINE_LEN = 1
MAX_LINE_LEN = 256
NUM_WORKERS = 0

EVAL_FREQUENCY = 1
EVAL_SAMPLE_COUNT = 8
EVAL_PREFIX_CHARS = 4

OUTPUT_PATH = Path('runs/decoder_lm_bytes.pt')


class DecoderLM(nn.Module):

    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout, pad_id, max_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.embed_scale = math.sqrt(d_model)
        self.pos = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, key_padding_mask):
        embedded = self.embed(input_ids) * self.embed_scale
        embedded = self.pos(embedded.transpose(0, 1))
        seq_len = embedded.size(0)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=embedded.device), diagonal=1
        )
        hidden = self.encoder(embedded, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return self.output(hidden)


class TextLineDataset(Dataset):

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)


def make_collate(pad_id):
    def collate(batch):
        inputs = [seq[:-1] for seq in batch]
        targets = [seq[1:] for seq in batch]
        max_len = max(seq.numel() for seq in inputs)
        batch_size = len(batch)
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        target_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        key_padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
        for i, (inp, tgt) in enumerate(zip(inputs, targets)):
            length = inp.numel()
            input_ids[i, :length] = inp
            target_ids[i, :length] = tgt
            key_padding_mask[i, :length] = False
        return input_ids, target_ids, key_padding_mask

    return collate


def _passes_filter(name: str) -> bool:
    name_lower = name.lower()
    if INCLUDE_DATA and not any(s.lower() in name_lower for s in INCLUDE_DATA):
        return False
    if EXCLUDE_DATA and any(s.lower() in name_lower for s in EXCLUDE_DATA):
        return False
    return True


def harvest_corpus(text_dirs: list[Path]) -> list[tuple[str, str]]:
    items = []
    for data_dir in text_dirs:
        root = Path(data_dir)
        candidates = {}
        for txt_path in sorted(root.rglob('*.txt')):
            name = txt_path.name
            if name.lower().endswith('.gt.txt'):
                stem_path = txt_path.with_name(name[:-len('.gt.txt')])
                is_plain = False
            else:
                stem_path = txt_path.with_name(name[:-len('.txt')])
                is_plain = True
            if not _passes_filter(name):
                continue
            key = str(stem_path)
            if key not in candidates or is_plain:
                candidates[key] = txt_path
        for txt_path in candidates.values():
            text = txt_path.read_text(encoding='utf-8').strip()
            if text:
                stratum = str(txt_path.relative_to(root).parent)
                items.append((text, stratum))
    return items


def harvest_external(external_files: list[Path]) -> list[str]:
    texts = []
    for file_path in external_files:
        for line in Path(file_path).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                texts.append(line)
    return texts


def stratified_split(items: list[tuple[str, str]], valid_split: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for text, stratum in items:
        groups[stratum].append(text)

    train_texts, valid_texts = [], []
    for texts in groups.values():
        rng.shuffle(texts)
        num_valid = int(len(texts) * valid_split)
        valid_texts.extend(texts[:num_valid])
        train_texts.extend(texts[num_valid:])

    rng.shuffle(train_texts)
    rng.shuffle(valid_texts)
    return train_texts, valid_texts


def build_sequences(texts: list[str], tokenizer: DeepTokenizer) -> list[list[int]]:
    sequences = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        if MIN_LINE_LEN + 2 <= len(ids) <= MAX_LINE_LEN:
            sequences.append(ids)
    return sequences


@torch.no_grad()
def evaluate(model, dataloader, vocab_size, pad_id, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for input_ids, target_ids, key_padding_mask in dataloader:
        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)
        key_padding_mask = key_padding_mask.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids, key_padding_mask)
            loss = F.cross_entropy(
                logits.to(torch.float32).permute(1, 0, 2).reshape(-1, vocab_size),
                target_ids.reshape(-1),
                ignore_index=pad_id,
                reduction='sum',
            )
        total_loss += loss.item()
        total_tokens += (target_ids != pad_id).sum().item()
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


@torch.no_grad()
def generate_samples(model, sequences, tokenizer, device, sample_count, prefix_chars, max_new):
    model.eval()
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id
    rows = []
    step = max(1, len(sequences) // sample_count)
    chosen = sequences[::step][:sample_count]
    for full_ids in chosen:
        target_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        prefix = [bos_id] + full_ids[1:1 + prefix_chars]
        generated = list(prefix)
        for _ in range(max_new):
            inp = torch.tensor([generated], dtype=torch.long, device=device)
            kpm = torch.zeros((1, len(generated)), dtype=torch.bool, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(inp, kpm)
            next_id = int(logits[-1, 0].argmax().item())
            if next_id == eos_id:
                break
            generated.append(next_id)
        prefix_text = tokenizer.decode(prefix, skip_special_tokens=True)
        generated_text = tokenizer.decode(generated, skip_special_tokens=True)
        rows.append(f"| `{prefix_text}` | `{generated_text}` | `{target_text}` |")
    header = "| prefix | generated | ground truth |\n| --- | --- | --- |\n"
    return header + "\n".join(rows)


def harvest_text_files(text_files: list[Path]) -> list[tuple[str, str]]:
    """Unlike EXTERNAL_TEXT_FILES, which is train-only, these lines join the stratified split, so
    checkpoint selection still has a validation loss when they are the only source."""
    items = []
    for file_path in text_files:
        stratum = Path(file_path).stem
        for line in Path(file_path).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                items.append((line, stratum))
    return items


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain the character decoder as a standalone LM.")
    p.add_argument('--text-dirs', nargs='*', type=Path, default=None,
                   help=f"Dirs to harvest *.txt from. Default: {[str(d) for d in TEXT_DATA_DIRS]}, "
                        f"which also holds data/valid's text. Pass no values to rely on --text-file.")
    p.add_argument('--text-file', nargs='*', type=Path, default=[],
                   help="Text files, one sample per line, split like the harvested text. "
                        "build_kenlm.py --dump-raw-text writes one without validation lines.")
    p.add_argument('--out', type=Path, default=None, help=f"Output path. Default: {OUTPUT_PATH}")
    return p.parse_args()


def main():
    args = parse_args()
    text_dirs = TEXT_DATA_DIRS if args.text_dirs is None else args.text_dirs
    output_path = args.out or OUTPUT_PATH

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("runs") / f'decoder_lm_{timestamp}'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    tokenizer = DeepTokenizer(normalize_unicode=True, verbose=True)
    pad_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    items = harvest_corpus(text_dirs)
    items.extend(harvest_text_files(args.text_file))
    num_strata = len({stratum for _, stratum in items})
    train_texts, valid_texts = stratified_split(items, VALID_SPLIT, SPLIT_SEED)
    train_texts.extend(harvest_external(EXTERNAL_TEXT_FILES))
    train_sequences = build_sequences(train_texts, tokenizer)
    valid_sequences = build_sequences(valid_texts, tokenizer)
    print(f"Harvested {len(items)} lines from {num_strata} source folders.")
    print(f"Train: {len(train_texts)} lines -> {len(train_sequences)} kept.")
    print(f"Valid: {len(valid_texts)} lines -> {len(valid_sequences)} kept (stratified {VALID_SPLIT:.0%} holdout).")
    if not train_sequences:
        print("No training text available. Check TEXT_DATA_DIRS and filters.")
        return

    collate = make_collate(pad_id)
    train_dataloader = DataLoader(
        TextLineDataset(train_sequences),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    valid_dataloader = DataLoader(
        TextLineDataset(valid_sequences),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    ) if valid_sequences else None

    model = DecoderLM(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=DECODER_HEADS,
        num_layers=DECODER_LAYERS,
        dim_feedforward=DECODER_FFN_DIM,
        dropout=DECODER_DROPOUT,
        pad_id=pad_id,
        max_len=MAX_SEQ_LEN,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Decoder LM parameters: {num_params / 1e6:.2f}M | vocab: {vocab_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True)

    def lr_lambda(step):
        return min(1.0, (step + 1) / WARMUP_STEPS)

    scheduler = LambdaLR(optimizer, lr_lambda)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        progress = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for input_ids, target_ids, key_padding_mask in progress:
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            key_padding_mask = key_padding_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(input_ids, key_padding_mask)
                loss = F.cross_entropy(
                    logits.to(torch.float32).permute(1, 0, 2).reshape(-1, vocab_size),
                    target_ids.reshape(-1),
                    ignore_index=pad_id,
                    label_smoothing=LABEL_SMOOTHING,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            num_target_tokens = (target_ids != pad_id).sum().item()
            total_loss += loss.item() * num_target_tokens
            total_tokens += num_target_tokens
            global_step += 1
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            writer.add_scalar('Learning_Rate/lm', scheduler.get_last_lr()[0], global_step)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(total_tokens, 1)
        train_perplexity = math.exp(min(avg_loss, 20.0))
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Perplexity/train', train_perplexity, epoch)
        print(f"Epoch {epoch + 1}: train loss {avg_loss:.4f} | perplexity {train_perplexity:.2f}")

        selection_loss = avg_loss
        if valid_dataloader is not None and (epoch + 1) % EVAL_FREQUENCY == 0:
            val_loss = evaluate(model, valid_dataloader, vocab_size, pad_id, device)
            val_perplexity = math.exp(min(val_loss, 20.0))
            writer.add_scalar('Loss/valid', val_loss, epoch)
            writer.add_scalar('Perplexity/valid', val_perplexity, epoch)
            print(f"Epoch {epoch + 1}: valid loss {val_loss:.4f} | perplexity {val_perplexity:.2f}")
            selection_loss = val_loss

            sample_source = valid_sequences if valid_sequences else train_sequences
            samples = generate_samples(
                model, sample_source, tokenizer, device, EVAL_SAMPLE_COUNT, EVAL_PREFIX_CHARS, MAX_LINE_LEN
            )
            writer.add_text('Samples/prefix_continuation', samples, epoch)

        if selection_loss < best_loss:
            best_loss = selection_loss
            torch.save({
                'lm_state_dict': model.state_dict(),
                'tokenizer_tokens': tokenizer.tokens_list,
                'config': {
                    'd_model': D_MODEL,
                    'decoder_layers': DECODER_LAYERS,
                    'decoder_heads': DECODER_HEADS,
                    'decoder_dim_feedforward': DECODER_FFN_DIM,
                    'decoder_dropout': DECODER_DROPOUT,
                },
                'epoch': epoch + 1,
                'loss': best_loss,
            }, output_path)
            print(f"Saved decoder LM to {output_path} (best loss {best_loss:.4f}).")

    writer.close()


if __name__ == "__main__":
    main()
