import json
import math
import multiprocessing as mp
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import albumentations as A
import torch
import torch.distributed as dist
from albumentations.pytorch import ToTensorV2
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.dataset import OCRDataset, WidthBucketedBatchSampler, make_collate_fn, worker_init
from utils.loss import CTCLossWithSmoothing
from utils.model import DeepOCR
from utils.tokenizer import DeepTokenizer, check_kenlm_artifacts

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# A multi-GPU run died mid-epoch in loss.backward() with a cuDNN fused-attention failure
# (mha_graph.execute(...).is_good() == false); keeping SDPA off cuDNN avoids that kernel.
torch.backends.cuda.enable_cudnn_sdp(False)

# 'cnn' (depthwise-separable) measured 1.6x slower than 'cnn_grouped' at similar size because the
# depthwise pair is memory-bound on a GPU. 'dinov3' and 'trocr' are why transformers is a dependency.
BACKBONE = 'cnn_grouped'
CNN_WIDTH_STRIDE = 8
FREEZE_BACKBONE = False
# 256 / 32 keeps eight feature rows, so the extra detail is spent in the backbone. kurrent_01, the
# worst dataset at 12.7% CER, arrives at a 245 px median and was losing detail to a 128 px resize.
PATCH_SIZE = 32
TARGET_HEIGHT = 256
# The dataset caps the aspect-preserved width here and squashes anything wider, so the cap has
# to scale with the height or doubling TARGET_HEIGHT silently distorts every long line.
MAX_WIDTH = 24 * TARGET_HEIGHT
USE_2D_STRUCTURE = False
WIDTH_STRIDE = CNN_WIDTH_STRIDE if BACKBONE in ('cnn', 'cnn_grouped') else PATCH_SIZE
PRETRAIN_WEIGHTS = None # Path to a best_model.pt to warm-start from; TRAIN_BASE_CHECKPOINT overrides it.
# '' = off, 'auto' = this run's dir or runs/latest_run if a last_checkpoint.pt exists, else a run
# dir or checkpoint path. Resuming reuses the original run dir and skips PRETRAIN_WEIGHTS.
RESUME_FROM = os.environ.get('OCR_RESUME', '')

# Only applies with USE_2D_STRUCTURE; 1D tokens are already one per column.
COLLAPSE_DEPTH = 3
COLLAPSE_HEIGHT_FOR_CTC = False
COLLAPSE_KERNEL_WIDTH = 3
DROP_PATH = 0.1
# curriculum: read mostly visible lines first, then hide more as the encoder learns to use context
MASK_RATIO = 0.1
MASK_RATIO_FINAL = 0.3
MOE_PENALTY = 0.05           # inert while ROUTER_BALANCE is 'loss_free'
NUM_ENCODER_LAYERS = 4
NUM_EXPERTS = int(os.environ.get('OCR_EXPERTS', 62)) # routed FFN experts; 1 = dense ablation
RESET_EXPERTS = False
RESET_MOE_ROUTER = False
ROUTER_Z_LOSS_COEF = 1e-3    # inert while ROUTER_BALANCE is 'loss_free'
TOP_K_EXPERTS = 8
EXPERT_FFN = 'swiglu'
EXPERT_HIDDEN = 352           # 62 x 352 at top-8 is the active width of 16 x 2048 at top-2
NUM_SHARED_EXPERTS = 1
ROUTER_BALANCE = 'loss_free'  # replaces MOE_PENALTY and the z-loss with a per-expert selection bias
ROUTER_BIAS_RATE = 1e-3
# Extra routed experts that convolve over neighbouring columns, kernel 15 = about four characters.
# Every FFN expert is pointwise; these are the only ones that see context. Need EXPERT_FFN 'swiglu'.
NUM_CONV_EXPERTS = 2
CONV_EXPERT_KERNEL = 15

# gt4histocr keeps long s on 68.8% of its lines and the oblique hyphen on 61.1% while other
# corpora fold both, so the convention is a per-line input. Written by scripts/audit_conventions.py;
# rerun it when sources change, since a corpus with no row trains as normalized.
CONVENTIONS_PATH = Path('data/conventions.json')
if CONVENTIONS_PATH.exists():
    _conventions = json.loads(CONVENTIONS_PATH.read_text(encoding='utf-8'))
    CONVENTION_NAMES = tuple(_conventions['names'])
    CONVENTION_BY_DATASET = _conventions['by_dataset']
else:
    CONVENTION_NAMES = ()
    CONVENTION_BY_DATASET = {}
WARMUP_ROUTER_EPOCHS = 0

USE_DECODER = True
CTC_LOSS_WEIGHT = 1.0
DECODER_DROPOUT = 0.1
DECODER_EOS_WEIGHT = 2.0
DECODER_FFN_DIM = 2048
DECODER_HEADS = 8
DECODER_LABEL_SMOOTHING = 0.1
DECODER_LAYERS = 4
DECODER_LOSS_WEIGHT = 1.0
DECODER_MAX_LEN = 256
DECODER_VAL_FREQUENCY = 1
DECODER_WARMUP_EPOCHS = 0
PRETRAINED_DECODER_LM = Path('runs/decoder_lm_bytes.pt')
SAVE_BEST_OF_HEADS = False
SELECT_ON_DECODER = False

BASE_BATCH_SIZE = 60
# Per GPU. On an H200 at the widest bucket (width 6144): 80 peaks at 127.5 of 140 GiB, 88 OOMs.
BATCH_SIZE = int(os.environ.get('OCR_BATCH', 80))
# A 100-epoch run was cut off by the job time limit around epoch 21 and the LR never annealed;
# pick a count the wall clock can finish.
EPOCHS = 60
BACKBONE_BASE_LR = 1e-4
CTC_ENTROPY_WEIGHT = 0.1
DECODER_BASE_LR = 1e-4
DISABLE_AUG_AFTER_EPOCH = None
EMA_DECAY = 0.999
MIN_LR = 1e-6
TRANSFORMER_BASE_LR = 1e-4
WARMUP_EPOCHS = 1.0
WEIGHT_DECAY = 1e-4

BEAM_VAL_FREQUENCY = 10
BEAM_WIDTH = 4
RESCORE_ALPHA = 0.3
RESCORE_BETA = 0.0
RESCORE_NBEST = 8
VAL_FREQUENCY = 1

BUCKET_SIZE_MULTIPLIER = 50
SAMPLER_SEED = 0
# Byte-level 6-gram over the 2,348,261 unfolded training lines. An LM built before the conventions
# were unfolded never saw a long s and steers beam search away from it. A dict that is not
# index-parallel to the tokenizer is refused and beam search runs without the LM.
KENLM_PATH = Path('data/lm/union_bytes_v2.bin')
LM_DICT_PATH = Path('data/lm/union_bytes_v2.dict')
LM_WEIGHT = 1.5 # swept on the byte base model over 1200 valid samples: 0.5->0.02991,
# 1.0->0.02929, 1.5->0.02881, 2.0->0.02942, 2.5->0.03097. Re-sweep when the LM or vocab changes.

CACHE_TO_RAM = False
EXCLUDE_DATA = []
IMAGENET_NORMALIZED_WHITE = (1.0 - 0.456) / 0.224
INCLUDE_DATA = []
# Per rank. Each spawn worker holds a copy of the dataset, so host RAM scales with GPUs x workers x
# prefetch: 7 GPUs x 16 x prefetch 4 OOM'd a 512G box (562G RSS); 8 keeps 7 GPUs fed.
NUM_WORKERS = int(os.environ.get('OCR_NUM_WORKERS', 8))
# The full 259k valid split takes hours to decode every epoch; a seeded 25k subset gives a stable CER.
VALID_LIMIT = int(os.environ.get('OCR_VALID_LIMIT', 25000))
# Per-step curves are unreadable at ~2,300 steps an epoch and dominate the event file, so the
# batch-level scalars are sampled. Epoch-level metrics are always written.
SCALAR_LOG_INTERVAL = int(os.environ.get('OCR_SCALAR_LOG_INTERVAL', 50))
PIN_MEMORY = True

TRAIN_DATA_DIR = Path('data/train')
VALID_DATA_DIR = Path('data/valid')
SPLIT_MANIFEST = Path('data/split_manifest.json')
# A short example; the full corpus list is in the training data section of README.md.
DATASET_SOURCES = [
    "data_sources/catmus_medieval",
    "data_sources/catmus_modern/pairs",
    "data_sources/gt4histocr/corpus/dta19",
    "data_sources/gt4histocr/corpus/RIDGES-Fraktur",
    "data_sources/NotarialDeeds_HTR",
    "data_sources/reichsanzeiger",
]

# Dataset labels filtered from the split at load time, for corpora whose ground truth cannot
# match line transcription (word-level or structure-token GT pushed per-dataset CER above 1.0).
DROP_DATASETS = []

DEBUG = False

if DEBUG:
    LIMIT = 200
else:
    # OCR_LIMIT caps samples per split for bisecting failures that only appear at scale.
    LIMIT = int(os.environ['OCR_LIMIT']) if os.environ.get('OCR_LIMIT') else None

# TRAIN_* overrides for runs launched by an external training app; no-ops when unset.
LABEL_OVERRIDES = {}
if os.environ.get('TRAIN_SNAPSHOT'):
    TRAIN_DATA_DIR = Path(os.environ['TRAIN_SNAPSHOT']) / 'train'
    VALID_DATA_DIR = Path(os.environ['TRAIN_SNAPSHOT']) / 'valid'
    SPLIT_MANIFEST = None
    DATASET_SOURCES = []
    # The snapshot maps every crop to the dataset it came from; tagging the valid
    # samples with it makes validation report per-dataset CER into progress.jsonl.
    _manifest = Path(os.environ['TRAIN_SNAPSHOT']) / 'manifest.json'
    if _manifest.is_file():
        LABEL_OVERRIDES = json.loads(_manifest.read_text(encoding='utf-8'))
if 'TRAIN_BASE_CHECKPOINT' in os.environ:
    PRETRAIN_WEIGHTS = Path(os.environ['TRAIN_BASE_CHECKPOINT']) if os.environ['TRAIN_BASE_CHECKPOINT'] else None
if os.environ.get('TRAIN_BATCH'):
    BATCH_SIZE = int(os.environ['TRAIN_BATCH'])
if os.environ.get('TRAIN_LR'):
    BACKBONE_BASE_LR = TRANSFORMER_BASE_LR = DECODER_BASE_LR = float(os.environ['TRAIN_LR'])

# Constants the app may tune per run, each read from TRAIN_<NAME> and parsed with the type
# of its default. Unset or empty keeps the default. A per-group TRAIN_*_BASE_LR wins over
# the shared TRAIN_LR above.
APP_TUNABLES = (
    'EPOCHS', 'WARMUP_EPOCHS', 'MIN_LR', 'WEIGHT_DECAY', 'EMA_DECAY',
    'BACKBONE_BASE_LR', 'TRANSFORMER_BASE_LR', 'DECODER_BASE_LR',
    'CTC_LOSS_WEIGHT', 'DECODER_LOSS_WEIGHT', 'CTC_ENTROPY_WEIGHT', 'DECODER_LABEL_SMOOTHING',
    'DECODER_EOS_WEIGHT', 'MOE_PENALTY', 'ROUTER_Z_LOSS_COEF',
    'EXPERT_FFN', 'EXPERT_HIDDEN', 'NUM_SHARED_EXPERTS', 'ROUTER_BALANCE', 'ROUTER_BIAS_RATE',
    'NUM_CONV_EXPERTS', 'CONV_EXPERT_KERNEL',
    'DROP_PATH', 'DECODER_DROPOUT', 'MASK_RATIO', 'MASK_RATIO_FINAL', 'DISABLE_AUG_AFTER_EPOCH',
    'VALID_LIMIT', 'VAL_FREQUENCY', 'BEAM_VAL_FREQUENCY', 'DECODER_VAL_FREQUENCY', 'BEAM_WIDTH',
    'SCALAR_LOG_INTERVAL',
    'SELECT_ON_DECODER', 'SAVE_BEST_OF_HEADS',
    'FREEZE_BACKBONE', 'RESET_EXPERTS', 'RESET_MOE_ROUTER', 'WARMUP_ROUTER_EPOCHS',
    'DECODER_WARMUP_EPOCHS', 'SAMPLER_SEED', 'NUM_WORKERS',
)


def parse_tunable(raw: str, default):
    if isinstance(default, bool):
        return raw.strip().lower() in ('1', 'true', 'yes', 'on')
    if isinstance(default, str):
        return raw.strip()
    if isinstance(default, int) or default is None:
        return int(raw)
    return float(raw)


for _name in APP_TUNABLES:
    if os.environ.get(f'TRAIN_{_name}', '') != '':
        globals()[_name] = parse_tunable(os.environ[f'TRAIN_{_name}'], globals()[_name])
if os.environ.get('TRAIN_SNAPSHOT'):
    # Small app snapshots must still give every DDP rank at least one full batch,
    # or the width-bucketed sampler shards to zero iterations per rank.
    _pairs = len(list(TRAIN_DATA_DIR.glob('*.txt')))
    _world = int(os.environ.get('WORLD_SIZE', '1'))
    if _pairs:
        BATCH_SIZE = max(1, min(BATCH_SIZE, _pairs // _world))

def _renamed(transform) -> bool:
    """albumentationsX 2.3 renamed the limits to `*_range` and only warns about the old names,
    so a limit spelled the other way silently falls back to the library default."""
    return 'brightness_range' in transform.InitSchema.model_fields


def _brightness_contrast(p: float = 1.0):
    if _renamed(A.RandomBrightnessContrast):
        return A.RandomBrightnessContrast(brightness_range=(-0.1, 0.1),
                                          contrast_range=(-0.1, 0.1), p=p)
    return A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                      contrast_limit=(-0.1, 0.1), p=p)


# A stack built on one albumentations install may be trained on another version.
RENAMED_LIMITS = {
    'ColorJitter': {'brightness': 'brightness_range', 'contrast': 'contrast_range',
                    'saturation': 'saturation_range', 'hue': 'hue_range'},
    'RandomBrightnessContrast': {'brightness_limit': 'brightness_range',
                                 'contrast_limit': 'contrast_range'},
}


def _respelled(node):
    """from_dict silently drops an argument the class does not take, which for hue means a full
    rotation instead of a tenth of one."""
    if isinstance(node, list):
        return [_respelled(item) for item in node]
    if not isinstance(node, dict):
        return node
    node = {key: _respelled(value) for key, value in node.items()}
    name = str(node.get('__class_fullname__', '')).rsplit('.', 1)[-1]
    aliases, transform = RENAMED_LIMITS.get(name), getattr(A, name, None)
    if not aliases or transform is None:
        return node
    takes = transform.InitSchema.model_fields
    either = {**aliases, **{new: old for old, new in aliases.items()}}
    respelled = {}
    for key, value in node.items():
        other = either.get(key)
        respelled[other if key not in takes and other in takes else key] = value
    return respelled


# Augmentations run after the resize to TARGET_HEIGHT, so anything measured in pixels has to
# track it or it weakens as the geometry grows. Measured median stroke width is 3.4 px at 128
# and 5.6 px at 256, so the scale factor keeps every pixel-absolute op at the same strength
# relative to the ink. Fractional transforms (noise, dropout, shadow, spatter, affine) need none.
_AUG_PIXEL_SCALE = TARGET_HEIGHT / 128


def _scaled(low: float, high: float) -> tuple[int, int]:
    return max(1, round(low * _AUG_PIXEL_SCALE)), max(1, round(high * _AUG_PIXEL_SCALE))


def get_transforms() -> tuple[A.Compose, A.Compose]:
    val_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])

    if os.environ.get('TRAIN_AUGMENT'):
        # Normalisation and the tensor conversion are appended so a saved stack cannot omit them.
        spec = json.loads(Path(os.environ['TRAIN_AUGMENT']).read_text(encoding='utf-8'))
        steps = list(A.from_dict(_respelled(spec)))
        return A.Compose([*steps, A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                          ToTensorV2()]), val_transform

    train_transform = A.Compose([
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent={
                'x': (-0.02, 0.02),
                'y': (-0.02, 0.02)
            },
            rotate=(-2, 2),
            shear={
                'x': (-2, 2),
                'y': (2, -2)
            },
            p=0.3,
            fill=255
        ),
        A.Affine(
            scale={'x': (0.7, 1.0), 'y': (1.0, 1.0)},
            keep_ratio=False,
            p=0.3,
            fill=255
        ),
        # Models slant and baseline drift. At 128 px alpha 1 was a no-op (CER 3.45 vs 3.40 clean),
        # 40 gave 3.9%, 200 destroyed the line (11.63%). approximate: 92.8 -> 16.7 ms per line.
        A.ElasticTransform(alpha=40 * _AUG_PIXEL_SCALE, sigma=50 * _AUG_PIXEL_SCALE,
                           approximate=True, p=0.2),
        # The default roi puts every shadow on the bottom half; full-line cost 3.55% vs 3.52% CER.
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.3),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(sigma_limit=(0.5 * _AUG_PIXEL_SCALE, 3.0 * _AUG_PIXEL_SCALE), blur_limit=0, p=0.3),
        # Dilation erases strokes past the stroke width: at 128 px scale 7 doubled CER to 7.59%.
        # Erosion only fattens the pen and stays harmless at 9 (4.05%).
        A.OneOf([
            A.Morphological(operation='erosion', scale=_scaled(2, 6), p=1.0),
            A.Morphological(operation='dilation', scale=_scaled(2, 4), p=1.0)
        ],
                p=0.3),
        # Every line is greyscale (R=G=B on 1100 sampled images), so channel ops measured 0.00 CER.
        _brightness_contrast(p=0.3),
        A.PixelDropout(dropout_prob=0.05, per_channel=True, p=0.3),
        A.CoarseDropout(num_holes_range=(1, 5), hole_height_range=(0.05, 0.15), hole_width_range=(0.02, 0.08), fill=0, p=0.3),
        A.Spatter(mode='mud', color=(0, 0, 0), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_transform, val_transform


def make_lr_lambda(base_lr: float, warmup_steps: int, total_steps: int, min_lr: float):
    def lr_lambda(step: int) -> float:
        # step + 1 because LambdaLR evaluates lambda(0) at construction; avoids a zero first LR
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        min_ratio = min_lr / base_lr
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def cleanup_empty_runs():
    runs_dir = Path("runs")
    if runs_dir.exists():
        for d in sorted(runs_dir.iterdir()):
            try:
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
            except (FileNotFoundError, OSError):
                pass


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        cleanup_empty_runs()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    is_master = local_rank == 0

    if is_distributed:
        # NCCL busy-waits at 100% GPU while blocked, so a hang looks like training; shorten the
        # watchdog with OCR_NCCL_TIMEOUT_MIN when bisecting one.
        nccl_timeout = int(os.environ.get('OCR_NCCL_TIMEOUT_MIN', 60))
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=nccl_timeout))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    effective_batch_size = BATCH_SIZE * world_size
    k_factor = max(1, effective_batch_size // BASE_BATCH_SIZE)

    dynamic_backbone_lr = BACKBONE_BASE_LR * math.sqrt(k_factor)
    dynamic_transformer_lr = TRANSFORMER_BASE_LR * math.sqrt(k_factor)
    dynamic_decoder_lr = DECODER_BASE_LR * math.sqrt(k_factor)

    timestamp = [datetime.now().strftime("%Y%m%d_%H%M%S")]
    if is_distributed:
        dist.broadcast_object_list(timestamp, src=0)

    if os.environ.get('TRAIN_OUTPUT'):
        log_dir = Path(os.environ['TRAIN_OUTPUT']) / 'run'
    else:
        log_dir = Path("runs") / f'ocr_{timestamp[0]}'

    resume_checkpoint = None
    if RESUME_FROM:
        if RESUME_FROM == 'auto':
            candidates = [log_dir / 'last_checkpoint.pt', Path('runs') / 'latest_run' / 'last_checkpoint.pt']
        else:
            target = Path(RESUME_FROM)
            candidates = [target / 'last_checkpoint.pt' if target.is_dir() else target]
        resume_target = next((c for c in candidates if c.exists()), None)
        if resume_target is not None:
            resume_checkpoint = resume_target.resolve()
            log_dir = resume_checkpoint.parent
        elif is_master:
            print(f"OCR_RESUME={RESUME_FROM}: nothing to resume; starting fresh.")

    if is_master:
        print(f"Using device: {device} | World Size: {world_size}")
        print(f"Per-GPU Batch: {BATCH_SIZE} | Effective Batch: {effective_batch_size}")
        print(f"Scaled Backbone LR: {dynamic_backbone_lr:.6f} | Scaled Transformer LR: {dynamic_transformer_lr:.6f}")
        log_dir.mkdir(parents=True, exist_ok=True)
        if log_dir.parent == Path('runs'):
            # latest_run always names the active run so OCR_RESUME=auto finds it after a requeue.
            latest_link = Path('runs') / 'latest_run'
            latest_link.unlink(missing_ok=True)
            latest_link.symlink_to(log_dir.name)

    train_transform, val_transform = get_transforms()
    # The .dict must stay index-parallel to the token inventory, so pair the LM only after the
    # tokenizer exists: build without one, verify, then rebuild with it if the dict lines up.
    probe = DeepTokenizer(normalize_unicode=True, verbose=False)
    kenlm_path, lm_dict_path = check_kenlm_artifacts(KENLM_PATH, LM_DICT_PATH, probe.tokens_list)
    tokenizer = DeepTokenizer(
        normalize_unicode=True,
        verbose=is_master,
        kenlm_path=kenlm_path,
        lm_dict_path=lm_dict_path,
        lm_weight=LM_WEIGHT,
    )

    train_dataset = OCRDataset(
        data_dir=TRAIN_DATA_DIR,
        tokenizer=tokenizer,
        augmentations=train_transform,
        no_aug_augmentations=val_transform,
        target_height=TARGET_HEIGHT,
        max_width=MAX_WIDTH,
        limit=LIMIT,
        patch_size=PATCH_SIZE,
        patch_size_w=WIDTH_STRIDE,
        use_2d_structure=USE_2D_STRUCTURE,
        collapse_height=COLLAPSE_HEIGHT_FOR_CTC,
        cache_to_ram=CACHE_TO_RAM,
        exclude_data=EXCLUDE_DATA,
        include_data=INCLUDE_DATA,
        manifest_path=SPLIT_MANIFEST,
        dataset_sources=DATASET_SOURCES,
        exclude_datasets=DROP_DATASETS
    )

    valid_dataset = OCRDataset(
        data_dir=VALID_DATA_DIR,
        tokenizer=tokenizer,
        augmentations=val_transform,
        target_height=TARGET_HEIGHT,
        max_width=MAX_WIDTH,
        limit=(LIMIT if LIMIT is not None else VALID_LIMIT),
        shuffle_before_limit=True,
        patch_size=PATCH_SIZE,
        patch_size_w=WIDTH_STRIDE,
        use_2d_structure=USE_2D_STRUCTURE,
        collapse_height=COLLAPSE_HEIGHT_FOR_CTC,
        cache_to_ram=CACHE_TO_RAM,
        exclude_data=EXCLUDE_DATA,
        include_data=INCLUDE_DATA,
        manifest_path=SPLIT_MANIFEST,
        dataset_sources=DATASET_SOURCES,
        exclude_datasets=DROP_DATASETS,
        label_overrides=LABEL_OVERRIDES
    )

    if is_master:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(valid_dataset)}")

    model = DeepOCR(
        tokenizer=tokenizer,
        backbone_name=BACKBONE,
        cnn_width_stride=CNN_WIDTH_STRIDE,
        cnn_patch_size=PATCH_SIZE,
        freeze_backbone=FREEZE_BACKBONE,
        target_height=TARGET_HEIGHT,
        weight_path=None if resume_checkpoint else PRETRAIN_WEIGHTS,
        use_2d_structure=USE_2D_STRUCTURE,
        num_experts=NUM_EXPERTS,
        top_k_experts=TOP_K_EXPERTS,
        moe_penalty=MOE_PENALTY,
        router_z_coef=ROUTER_Z_LOSS_COEF,
        expert_ffn=EXPERT_FFN,
        expert_hidden=EXPERT_HIDDEN,
        num_shared_experts=NUM_SHARED_EXPERTS,
        router_balance=ROUTER_BALANCE,
        router_bias_rate=ROUTER_BIAS_RATE,
        num_conv_experts=NUM_CONV_EXPERTS,
        conv_expert_kernel=CONV_EXPERT_KERNEL,
        convention_names=CONVENTION_NAMES,
        convention_by_dataset=CONVENTION_BY_DATASET,
        mask_ratio=MASK_RATIO,
        max_width=MAX_WIDTH,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        drop_path=DROP_PATH,
        collapse_height_for_ctc=COLLAPSE_HEIGHT_FOR_CTC,
        collapse_kernel_width=COLLAPSE_KERNEL_WIDTH,
        collapse_depth=COLLAPSE_DEPTH,
        use_decoder=USE_DECODER,
        decoder_layers=DECODER_LAYERS,
        decoder_heads=DECODER_HEADS,
        decoder_dim_feedforward=DECODER_FFN_DIM,
        decoder_dropout=DECODER_DROPOUT,
        ctc_loss_weight=CTC_LOSS_WEIGHT,
        decoder_loss_weight=DECODER_LOSS_WEIGHT,
        decoder_label_smoothing=DECODER_LABEL_SMOOTHING,
        decoder_eos_weight=DECODER_EOS_WEIGHT,
        decoder_lm_path=PRETRAINED_DECODER_LM
    ).to(device)

    if is_distributed:
        model = cast(DeepOCR, torch.nn.SyncBatchNorm.convert_sync_batchnorm(model))

    if is_master:
        model.print_parameter_summary()

    if is_master and RESET_MOE_ROUTER:
        print("Resetting MoE router gates to break checkpoint bias...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'gate' in name and RESET_MOE_ROUTER:
            torch.nn.init.normal_(module.weight, mean=0, std=0.01)

    if is_master and RESET_EXPERTS:
        print("Resetting MoE expert weights to break checkpoint bias...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'expert' in name and RESET_EXPERTS:
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')

    collate = make_collate_fn(
        patch_size=model.patch_size,
        patch_size_w=model.patch_size_w,
        use_2d_structure=USE_2D_STRUCTURE,
        fill_value=IMAGENET_NORMALIZED_WHITE,
        collapse_height=COLLAPSE_HEIGHT_FOR_CTC
    )

    train_batch_sampler = WidthBucketedBatchSampler(
        widths=train_dataset.sample_widths,
        batch_size=BATCH_SIZE,
        num_replicas=world_size,
        rank=dist.get_rank() if is_distributed else 0,
        shuffle=True,
        seed=SAMPLER_SEED,
        bucket_size_multiplier=BUCKET_SIZE_MULTIPLIER,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        multiprocessing_context='spawn' if NUM_WORKERS > 0 else None,
        worker_init_fn=worker_init,
    )

    valid_sampler = DistributedSampler(valid_dataset, shuffle=False) if is_distributed else None

    # Respawning workers took 41 of a 48 s validation pass on 2 x H200; they are alive during
    # validation anyway, so persisting them adds nothing to peak host memory.
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=valid_sampler,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        multiprocessing_context='spawn' if NUM_WORKERS > 0 else None,
        worker_init_fn=worker_init,
    )

    backbone_params, decoder_params, transformer_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('backbone.'):
            backbone_params.append(param)
        elif name.startswith('decoder'):
            decoder_params.append(param)
        else:
            transformer_params.append(param)

    param_groups = [{
        'params': backbone_params,
        'lr': dynamic_backbone_lr,
        'weight_decay': WEIGHT_DECAY
    }, {
        'params': transformer_params,
        'lr': dynamic_transformer_lr,
        'weight_decay': WEIGHT_DECAY
    }]
    if decoder_params:
        param_groups.append({
            'params': decoder_params,
            'lr': dynamic_decoder_lr,
            'weight_decay': WEIGHT_DECAY
        })
        if is_master:
            print(f"Decoder param group: {len(decoder_params)} tensors at LR {dynamic_decoder_lr:.6f}")

    optimizer = torch.optim.AdamW(param_groups, fused=True)

    steps_per_epoch = len(train_dataloader)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = int(WARMUP_EPOCHS * steps_per_epoch)
    scheduler = LambdaLR(optimizer, lr_lambda=[
        make_lr_lambda(group['lr'], warmup_steps, total_steps, MIN_LR) for group in optimizer.param_groups
    ])
    criterion = CTCLossWithSmoothing(blank_id=tokenizer.blank_token_id, smoothing_weight=CTC_ENTROPY_WEIGHT, zero_infinity=True)

    # Stacked expert weights mean no parameter is ever unused, so static_graph is safe:
    # 805.6 -> 790.2 ms a step on 2 x RTX PRO 6000.
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        static_graph=True,
    ) if is_distributed else None

    best_cer = model.train_model( #type: ignore
        train_dataloader,
        valid_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=EPOCHS,
        scheduler=scheduler,
        log_dir=log_dir,
        beam_width=BEAM_WIDTH,
        debug=DEBUG,
        val_frequency=VAL_FREQUENCY,
        router_warmup_epochs=WARMUP_ROUTER_EPOCHS,
        beam_val_frequency=BEAM_VAL_FREQUENCY,
        profile_mode=False,
        ddp_model=ddp_model,
        train_sampler=train_batch_sampler,
        is_master=is_master,
        world_size=world_size,
        disable_aug_after_epoch=DISABLE_AUG_AFTER_EPOCH,
        decoder_warmup_epochs=DECODER_WARMUP_EPOCHS,
        decoder_val_frequency=DECODER_VAL_FREQUENCY,
        decoder_max_len=DECODER_MAX_LEN,
        select_on_decoder=SELECT_ON_DECODER,
        save_best_of_heads=SAVE_BEST_OF_HEADS,
        ema_decay=EMA_DECAY,
        mask_ratio_final=MASK_RATIO_FINAL,
        rescore_alpha=RESCORE_ALPHA,
        rescore_beta=RESCORE_BETA,
        rescore_nbest=RESCORE_NBEST,
        resume_checkpoint=resume_checkpoint,
        scalar_log_interval=SCALAR_LOG_INTERVAL
    )

    if is_master and os.environ.get('TRAIN_OUTPUT'):
        out = Path(os.environ['TRAIN_OUTPUT'])
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(log_dir / 'best_model.pt', out / 'model.ckpt')
        (out / 'metrics.json').write_text(json.dumps({'cer': best_cer}))

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
