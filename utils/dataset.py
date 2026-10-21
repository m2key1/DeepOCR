import os
import json
import math
import random
import pickle
import hashlib
import concurrent.futures
from pathlib import Path
from typing import Iterator, Optional

import cv2
import torch
import numpy as np
from PIL import Image
import albumentations as A
import torchvision.transforms.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from utils.tokenizer import DeepTokenizer, canonicalize


def worker_init(worker_id: int) -> None:
    # OpenCV's per-process thread pool multiplies across ranks x workers, and the augmentation work
    # does not parallelise, so the threads are pure contention.
    cv2.setNumThreads(0)


def _sanitize_label(token: str) -> str:
    cleaned = token.replace('data_sources/', '').strip('/')
    return cleaned.replace('/', '_') or 'unknown'

# Leaf names that say nothing alone: catmus_modern/pairs would show in TensorBoard as "pairs",
# and a source named "train" reads as the training split. These get qualified with the parent.
_AMBIGUOUS_LEAVES = {'train', 'valid', 'test', 'full', 'pairs', 'data', 'images', 'lines', 'corpus'}


def derive_dataset_label(source_path: str, dataset_sources: list[str]) -> str:
    """Maps a sample's source path to a dataset label. A dataset_sources entry matches on the whole
    token or on its leaf as a path component (PAGE XML crops live under _page_lines/<leaf>/), and
    the leaf becomes the label. Falls back to the path segments after 'data_sources'."""
    normalized = source_path.replace('\\', '/')
    for src in dataset_sources:
        token = src.replace('\\', '/').strip('/')
        if not token:
            continue
        parts = token.split('/')
        leaf = parts[-1]
        if token in normalized or f'/{leaf}/' in normalized:
            if leaf.lower() in _AMBIGUOUS_LEAVES and len(parts) >= 2:
                return _sanitize_label('/'.join(parts[-2:]))
            return _sanitize_label(leaf)
    parts = [p for p in normalized.split('/') if p]
    if 'data_sources' in parts:
        tail = parts[parts.index('data_sources') + 1:-1]
        if tail:
            return _sanitize_label('/'.join(tail[:2]))
    return _sanitize_label(parts[-2]) if len(parts) >= 2 else 'unknown'

def _get_prominent_color_from_pil(img: Image.Image) -> tuple[int, int, int]:
    small_img = img.resize((32, 32), Image.Resampling.NEAREST)
    small_array = np.array(small_img)
    
    if small_array.shape[-1] == 4:
        small_alpha = small_array[:, :, 3]
        valid_pixels = small_array[small_alpha > 128][:, :3]
    else:
        valid_pixels = small_array.reshape(-1, 3)

    if len(valid_pixels) == 0:
        return (255, 255, 255)
        
    colors, counts = np.unique(valid_pixels, axis=0, return_counts=True)
    return tuple(colors[counts.argmax()])

def fill_transparency_with_prominent_color(image_path):
    img = Image.open(image_path)
    if img.mode != 'RGBA':
        if 'transparency' not in img.info:
            return img.convert('RGB')
    
    img = img.convert('RGBA')
    img_array = np.array(img)
    alpha = img_array[:, :, 3]

    if np.min(alpha) == 255:
        return img.convert('RGB')

    prominent_color = _get_prominent_color_from_pil(img)
    background = Image.new('RGB', img.size, prominent_color)
    background.paste(img, (0, 0), mask=img.split()[3])

    return background

def _process_dataset_item(kwargs):
    image_path = kwargs['image_path']
    text_path = kwargs['text_path']
    cache_to_ram = kwargs['cache_to_ram']
    target_height = kwargs['target_height']
    max_width = kwargs['max_width']
    patch_size = kwargs['patch_size']
    patch_size_w = kwargs.get('patch_size_w', patch_size)
    use_2d_structure = kwargs['use_2d_structure']
    collapse_height = kwargs.get('collapse_height', False)

    try:
        text = text_path.read_text(encoding='utf-8').strip()

        # Must match tokenizer.encode() exactly, which canonicalizes before encoding.
        text = canonicalize(text)

        if not text:
            return None, 'empty_text'

        # UTF-8 bytes bound the token count from above in both tokenizer modes.
        token_length = len(text.encode('utf-8'))

        if cache_to_ram:
            image = fill_transparency_with_prominent_color(image_path)
            w, h = image.size
        else:
            # Header-only read; no pixel decode needed to know the final width.
            with Image.open(image_path) as img:
                w, h = img.size

        new_width = int(w * (target_height / h))
        if max_width and new_width > max_width:
            new_width = max_width

        num_patches_w = new_width // patch_size_w
        num_patches_h = target_height // patch_size
        full_grid_capacity = use_2d_structure and not collapse_height
        total_capacity = (num_patches_w * num_patches_h) if full_grid_capacity else num_patches_w

        if total_capacity < token_length:
            return None, 'capacity_error'

        if cache_to_ram:
            image = image.resize((new_width, target_height), Image.Resampling.LANCZOS)
            return (np.array(image), text, new_width), 'success'
        return (str(image_path), text, new_width), 'success'

    except Exception:
        return None, 'error'


class OCRDataset(Dataset):
    def __init__(
        self,
        data_dir,
        tokenizer: DeepTokenizer,
        augmentations=None,
        no_aug_augmentations=None,
        image_extensions=('.png', '.jpg', '.jpeg'),
        limit: Optional[int] = None,
        shuffle_before_limit: bool = False,
        target_height=64,
        max_width: Optional[int] = None,
        patch_size: int = 16,
        patch_size_w: Optional[int] = None,
        use_2d_structure: bool = True,
        collapse_height: bool = False,
        cache_to_ram: bool = False,
        exclude_data: list[str] = [],
        include_data: list[str] = [],
        manifest_path: Optional[Path] = None,
        dataset_sources: list[str] = [],
        exclude_datasets: list[str] = [],
        label_overrides: Optional[dict] = None
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.image_extensions = image_extensions
        self.limit = limit
        self.shuffle_before_limit = shuffle_before_limit
        self.target_height = target_height
        self.max_width = max_width
        self.patch_size = patch_size
        self.patch_size_w = patch_size_w if patch_size_w is not None else patch_size
        self.use_2d_structure = use_2d_structure
        self.collapse_height = collapse_height
        self.cache_to_ram = cache_to_ram
        self.exclude_data = exclude_data
        self.include_data = include_data
        self.dataset_sources = dataset_sources
        # Exact path-component match on the manifest source path, so an id like '4095301' cannot
        # collide with a digit run elsewhere in the path.
        self.exclude_datasets = set(exclude_datasets)
        # stem -> label, for snapshots that tag each crop with its dataset for per-dataset CER
        self.label_overrides = label_overrides or {}
        self.stem_to_source = self._load_manifest_sources(manifest_path)

        if augmentations is None:
            self.augmentations = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.augmentations = augmentations

        self.no_aug_augmentations = no_aug_augmentations
        # Shared across DataLoader worker processes so toggling augmentation mid-training
        # propagates. A torch shared tensor rather than mp.Value because spawn-context workers
        # pickle the dataset, and mp.Value wraps an mmap that cannot be pickled.
        self._augment_enabled = torch.ones((), dtype=torch.uint8).share_memory_()

        self.data_pairs = self._find_pairs()

        if not self.data_pairs:
            raise FileNotFoundError(f"No valid image-text pairs found in directory: {self.data_dir}")

    def _load_manifest_sources(self, manifest_path: Optional[Path]) -> dict:
        if manifest_path is None:
            return {}
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            print(f"Split manifest not found at {manifest_path}; dataset labels will be path-derived.")
            return {}
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        sources = {}
        for split in ('train', 'valid'):
            sources.update(manifest.get(split, {}))
        return sources

    def _label_for(self, image_path: Path) -> str:
        override = self.label_overrides.get(image_path.stem)
        if override is not None:
            return override
        source = self.stem_to_source.get(image_path.stem, str(image_path))
        return derive_dataset_label(source, self.dataset_sources)

    def _pairs_cache_key(self) -> str:
        """Hashes every input that affects (pairs, labels, widths). Stat-ing every file is too
        slow, so the split directory's own mtime and size stand in for its contents."""
        try:
            st = os.stat(self.data_dir)
            data_sig = [st.st_mtime_ns, st.st_size]
        except OSError:
            data_sig = None
        payload = {
            'v': 1,
            'data_dir': str(Path(self.data_dir).resolve()),
            'data_sig': data_sig,
            'image_extensions': sorted(self.image_extensions),
            'include_data': sorted(self.include_data),
            'exclude_data': sorted(self.exclude_data),
            'exclude_datasets': sorted(self.exclude_datasets),
            'limit': self.limit,
            'shuffle_before_limit': self.shuffle_before_limit,
            'target_height': self.target_height,
            'max_width': self.max_width,
            'patch_size': self.patch_size,
            'patch_size_w': self.patch_size_w,
            'use_2d_structure': self.use_2d_structure,
            'collapse_height': self.collapse_height,
            'label_overrides': sorted(self.label_overrides.items()),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
        return hashlib.sha1(blob).hexdigest()[:16]

    def _pairs_cache_path(self) -> Optional[Path]:
        # Skip for cache_to_ram (payload would be full decoded images) or via OCR_PAIR_CACHE=0.
        if self.cache_to_ram or os.environ.get('OCR_PAIR_CACHE', '1') == '0':
            return None
        cache_dir = Path(self.data_dir).parent / '.pair_cache'
        return cache_dir / f"{Path(self.data_dir).name}_{self._pairs_cache_key()}.pkl"

    def _load_pairs_cache(self):
        path = self._pairs_cache_path()
        if path is None or not path.exists():
            return None
        try:
            with open(path, 'rb') as f:
                blob = pickle.load(f)
            return blob['data_pairs'], blob['sample_labels'], blob['sample_widths']
        except Exception as e:
            print(f"Ignoring unreadable pair cache {path}: {e}")
            return None

    def _save_pairs_cache(self, data_pairs, sample_labels, sample_widths):
        path = self._pairs_cache_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic publish: temp-per-pid then rename. Concurrent DDP ranks that all miss on a cold
            # start each write identical content, so last-writer-wins is harmless.
            tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
            with open(tmp, 'wb') as f:
                pickle.dump(
                    {'data_pairs': data_pairs, 'sample_labels': sample_labels,
                     'sample_widths': sample_widths},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except Exception as e:
            print(f"Could not write pair cache {path}: {e}")

    def _find_pairs(self):
        cached = self._load_pairs_cache()
        if cached is not None:
            data_pairs, self.sample_labels, self.sample_widths = cached
            print(f"Loaded {len(data_pairs)} image-text pairs from cache "
                  f"({self._pairs_cache_path().name}); skipped the image-header scan.")
            return data_pairs

        valid_paths = []
        filtered_count = 0
        for image_path in sorted(self.data_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in self.image_extensions:
                name_lower = image_path.name.lower()

                text_candidates = [
                    image_path.parent / (image_path.stem + ext)
                    for ext in ('.txt', '.gt.txt', '.corrected.txt')
                ]
                
                existing_texts = [path for path in text_candidates if path.exists()]
                if not existing_texts:
                    continue

                selected_text_path = None
                for text_path in existing_texts:
                    combined_name = f"{name_lower} {text_path.name.lower()}"
                    if self.include_data and not any(s.lower() in combined_name for s in self.include_data):
                        continue
                    if self.exclude_data and any(s.lower() in combined_name for s in self.exclude_data):
                        continue
                    selected_text_path = text_path
                    break

                if selected_text_path is None:
                    filtered_count += 1
                    continue

                if self.exclude_datasets:
                    source = self.stem_to_source.get(image_path.stem, str(image_path))
                    if self.exclude_datasets & set(source.replace('\\', '/').split('/')):
                        filtered_count += 1
                        continue

                valid_paths.append((image_path, selected_text_path))
                # Without the early stop a limited run still walks every entry. A shuffled
                # limit needs the full list to draw a subset across datasets.
                if self.limit and not self.shuffle_before_limit and len(valid_paths) >= self.limit:
                    break

        if self.limit:
            if self.shuffle_before_limit:
                random.Random(42).shuffle(valid_paths)
            valid_paths = valid_paths[:self.limit]
            
        tasks = []

        for img_path, txt_path in valid_paths:
            tasks.append({
                'image_path': img_path,
                'text_path': txt_path,
                'cache_to_ram': self.cache_to_ram,
                'target_height': self.target_height,
                'max_width': self.max_width,
                'patch_size': self.patch_size,
                'patch_size_w': self.patch_size_w,
                'use_2d_structure': self.use_2d_structure,
                'collapse_height': self.collapse_height,
            })

        data_pairs = []
        sample_labels = []
        sample_widths = []
        skipped_empty = 0
        skipped_capacity = 0

        num_workers = min(32, os.cpu_count() or 1)
        print(f"Found {len(valid_paths)} valid image-text pairs. Filtered out {filtered_count} pairs based on filter criteria.")
        print(f"Initializing dataset using {num_workers} CPU cores...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(_process_dataset_item, tasks), total=len(tasks), desc="Processing/Caching"))

        for (res, status), (src_image_path, _) in zip(results, valid_paths):
            if status == 'success' and res is not None:
                if self.cache_to_ram:
                    image_array, text, width = res
                    target = self.tokenizer.encode(text)
                    data_pairs.append((image_array, torch.tensor(target, dtype=torch.long), text))
                else:
                    image_path, text, width = res
                    data_pairs.append((image_path, text))
                sample_labels.append(self._label_for(src_image_path))
                sample_widths.append(width)

            elif status == 'empty_text':
                skipped_empty += 1
            elif status == 'capacity_error':
                skipped_capacity += 1

        if skipped_empty > 0 or skipped_capacity > 0:
            print(f"Dataset Loaded. Skipped: {skipped_empty} (empty text), {skipped_capacity} (too long).")

        self.sample_labels = sample_labels
        self.sample_widths = sample_widths
        label_counts = {label: sample_labels.count(label) for label in sorted(set(sample_labels))}
        print(f"Dataset composition by source: {label_counts}")

        self._save_pairs_cache(data_pairs, sample_labels, sample_widths)
        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def set_augmentation_enabled(self, enabled: bool):
        """Toggles training augmentation on/off; propagates to active DataLoader workers."""
        self._augment_enabled.fill_(1 if enabled else 0)

    def _active_augmentations(self):
        if self._augment_enabled.item() == 0 and self.no_aug_augmentations is not None:
            return self.no_aug_augmentations
        return self.augmentations

    def __getitem__(self, idx):
        augmentations = self._active_augmentations()
        label = self.sample_labels[idx]
        if self.cache_to_ram:
            image, target_tensor, _ = self.data_pairs[idx]
            if augmentations:
                image = augmentations(image=image)['image']
            return image, target_tensor, label

        image_path, text = self.data_pairs[idx]
        try:
            image = fill_transparency_with_prominent_color(image_path)
        except (OSError, ValueError) as e:
            print(f"Skipping unreadable image {image_path}: {e}")
            return None

        w, h = image.size
        new_width = int(w * (self.target_height / h))
        if self.max_width and new_width > self.max_width:
            new_width = self.max_width

        num_patches_w = new_width // self.patch_size_w
        num_patches_h = self.target_height // self.patch_size
        full_grid_capacity = self.use_2d_structure and not self.collapse_height
        total_capacity = (num_patches_w * num_patches_h) if full_grid_capacity else num_patches_w

        target = self.tokenizer.encode(text)
        if total_capacity < len(target):
            return None

        image = image.resize((new_width, self.target_height), Image.Resampling.LANCZOS)
        image = np.array(image)

        if augmentations:
            image = augmentations(image=image)['image']

        return image, torch.tensor(target, dtype=torch.long), label

class WidthBucketedBatchSampler(Sampler[list[int]]):
    """Batches samples of similar width together so per-batch padding stays tight. Each epoch,
    a seeded shuffle is chunked into windows, each window is sorted by width and sliced into
    batches, and the batch list is shuffled and sharded evenly across distributed ranks. All
    ranks derive the identical batch list from seed + epoch, so per-rank batch counts match."""

    def __init__(
        self,
        widths: list[int],
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        bucket_size_multiplier: int = 50,
    ):
        self.widths = torch.tensor(widths, dtype=torch.long)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = batch_size * bucket_size_multiplier
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(len(self.widths) / self.batch_size) // self.num_replicas

    def __iter__(self) -> Iterator[list[int]]:
        n = len(self.widths)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            order = torch.randperm(n, generator=generator)
        else:
            order = torch.arange(n)

        batches = []
        for start in range(0, n, self.window_size):
            window = order[start:start + self.window_size]
            window = window[torch.argsort(self.widths[window], stable=True)]
            for batch_start in range(0, len(window), self.batch_size):
                batches.append(window[batch_start:batch_start + self.batch_size].tolist())

        if self.shuffle:
            batch_order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in batch_order]

        usable = (len(batches) // self.num_replicas) * self.num_replicas
        for i in range(self.rank, usable, self.num_replicas):
            yield batches[i]


class Collate:
    """Pads a batch to a common size and builds the CTC length and mask tensors.

    A class rather than a closure: spawn-context dataloader workers pickle the collate function,
    and a function defined inside another function cannot be pickled.
    """

    def __init__(
        self,
        patch_size: int,
        patch_size_w: Optional[int] = None,
        use_2d_structure: bool = True,
        fill_value: float = 1.0,
        pad_width_to_multiple: bool = True,
        pad_height_to_multiple: bool = False,
        fixed_width: Optional[int] = None,
        collapse_height: bool = False
    ):
        self.patch_size = patch_size
        self.patch_size_w = patch_size_w if patch_size_w is not None else patch_size
        self.use_2d_structure = use_2d_structure
        self.fill_value = fill_value
        self.pad_width_to_multiple = pad_width_to_multiple
        self.pad_height_to_multiple = pad_height_to_multiple
        self.fixed_width = fixed_width
        self.collapse_height = collapse_height

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        images, targets, dataset_labels = zip(*batch)
        B = len(images)
        C = images[0].shape[0]

        heights = {img.shape[1] for img in images}
        if len(heights) > 1:
            raise ValueError(f"Images in batch have inconsistent heights: {heights}")
        H = next(iter(heights))
        
        if self.pad_height_to_multiple and (H % self.patch_size != 0):
            H_pad = math.ceil(H / self.patch_size) * self.patch_size - H
            images = [F.pad(img, (0, 0, 0, H_pad), mode="constant", value=self.fill_value) for img in images]
            H += H_pad
        if H % self.patch_size != 0:
            raise ValueError(f"Image height {H} must be divisible by self.patch_size {self.patch_size}.")

        widths = [img.shape[2] for img in images]

        if self.fixed_width is not None:
            batch_target_W = self.fixed_width
        else:
            batch_target_W = max(math.ceil(w / self.patch_size_w) for w in widths) * self.patch_size_w

        num_patches_h = H // self.patch_size
        num_patches_w = batch_target_W // self.patch_size_w

        batched_images = torch.full((B, C, H, batch_target_W), self.fill_value, dtype=images[0].dtype)
        
        if self.use_2d_structure:
            batched_masks = torch.ones((B, num_patches_w * num_patches_h), dtype=torch.bool)
        else:
            batched_masks = torch.ones((B, num_patches_w), dtype=torch.bool)
            
        input_lengths = torch.zeros(B, dtype=torch.long)

        for i, (img, w) in enumerate(zip(images, widths)):
            batched_images[i, :, :, :w] = img

            targ_w = math.ceil(w / self.patch_size_w) * self.patch_size_w if self.pad_width_to_multiple else w
            real_w_patches = targ_w // self.patch_size_w

            if self.use_2d_structure:
                valid_len = real_w_patches * num_patches_h
                batched_masks[i, :valid_len] = False
                input_lengths[i] = real_w_patches if self.collapse_height else valid_len
            else:
                batched_masks[i, :real_w_patches] = False
                input_lengths[i] = real_w_patches

        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets_concat = torch.cat([torch.as_tensor(t, dtype=torch.long) for t in targets], dim=0)

        long_idx = (target_lengths > input_lengths).nonzero(as_tuple=False).flatten().tolist()
        if long_idx:
            print(f"WARNING: Target length > model output length for samples {long_idx}")

        return batched_images, batched_masks, targets_concat, input_lengths, target_lengths, list(dataset_labels)


def make_collate_fn(
    patch_size: int,
    patch_size_w: Optional[int] = None,
    use_2d_structure: bool = True,
    fill_value: float = 1.0,
    pad_width_to_multiple: bool = True,
    pad_height_to_multiple: bool = False,
    fixed_width: Optional[int] = None,
    collapse_height: bool = False
):
    return Collate(patch_size, patch_size_w, use_2d_structure, fill_value,
                   pad_width_to_multiple, pad_height_to_multiple, fixed_width, collapse_height)
