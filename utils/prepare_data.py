import hashlib
import json
import math
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Literal, Optional

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from label_studio_sdk import Client
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.tokenizer import canonicalize

load_dotenv(".env")

MIN_LINE_HEIGHT = 8
IMAGE_LINK_MODE = 'hardlink'    # 'hardlink' | 'symlink' | 'copy'


def _place_image(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == 'hardlink':
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    elif mode == 'symlink':
        try:
            os.symlink(os.path.abspath(src), dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _rect_to_polygon(value: dict, page_w: int, page_h: int) -> list[tuple[int, int]] | None:
    x_pct = value.get('x')
    y_pct = value.get('y')
    w_pct = value.get('width')
    h_pct = value.get('height')
    if x_pct is None or y_pct is None or w_pct is None or h_pct is None:
        return None
    rotation = float(value.get('rotation', 0) or 0)
    x = x_pct / 100.0 * page_w
    y = y_pct / 100.0 * page_h
    w = w_pct / 100.0 * page_w
    h = h_pct / 100.0 * page_h
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    if rotation:
        rad = math.radians(rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        rotated = []
        for px, py in corners:
            dx, dy = px - x, py - y
            rotated.append((x + dx * cos_r - dy * sin_r, y + dx * sin_r + dy * cos_r))
        corners = rotated
    return [(int(round(px)), int(round(py))) for px, py in corners]


def _poly_to_polygon(value: dict, page_w: int, page_h: int) -> list[tuple[int, int]] | None:
    points = value.get('points')
    if not points:
        return None
    return [(int(round(px / 100.0 * page_w)), int(round(py / 100.0 * page_h))) for px, py in points]


def _crop_polygon_aabb(page_img: Image.Image, polygon: list[tuple[int, int]]) -> Image.Image | None:
    """Axis-aligned box with the polygon as an alpha mask. Used for PREP_CROP=aabb and when
    the min-rect crop raises."""
    page_w, page_h = page_img.size
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0 = max(0, min(xs))
    y0 = max(0, min(ys))
    x1 = min(page_w, max(xs))
    y1 = min(page_h, max(ys))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    shifted = [(p[0] - x0, p[1] - y0) for p in polygon]
    mask = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(mask).polygon(shifted, fill=255)
    crop = page_img.crop((x0, y0, x1, y1)).convert("RGBA")
    crop.putalpha(mask)
    return crop


def _crop_polygon_minrect(page_img: Image.Image, polygon: list[tuple[int, int]],
                          pad: int = 2) -> Image.Image | None:
    """Crops the polygon's minimum-area rectangle, deskewed to horizontal.

    Every crop is resized to a fixed height, so a skewed line's taller axis-aligned box costs
    glyph resolution: over 1,200 kurrent_01 pages AABB / min-rect height is 1.148 at the median
    and 1.968 at p90. Only a padded patch around the rectangle is rotated, not the page."""
    pts = np.asarray(polygon, dtype=np.float32)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    if w < h:                       # make the long side horizontal
        w, h = h, w
        angle += 90.0
    if w < 4 or h < 4:
        return None

    reach = int(math.ceil(math.hypot(w, h) / 2)) + pad + 2
    px0 = max(0, int(cx) - reach)
    py0 = max(0, int(cy) - reach)
    px1 = min(page_img.width, int(cx) + reach)
    py1 = min(page_img.height, int(cy) + reach)
    if px1 - px0 < 4 or py1 - py0 < 4:
        return None
    patch = np.asarray(page_img.crop((px0, py0, px1, py1)).convert('RGB'))

    matrix = cv2.getRotationMatrix2D((float(cx - px0), float(cy - py0)), angle, 1.0)
    rotated = cv2.warpAffine(patch, matrix, (patch.shape[1], patch.shape[0]),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    out_w = int(round(w)) + 2 * pad
    out_h = int(round(h)) + 2 * pad
    box = cv2.getRectSubPix(rotated, (out_w, out_h),
                            (float(cx - px0), float(cy - py0)))
    if box is None or box.shape[0] < 4 or box.shape[1] < 4:
        return None
    return Image.fromarray(box)


# 'minrect' (deskewed, default) or 'aabb' (axis-aligned with a polygon mask)
CROP_MODE = os.environ.get('PREP_CROP', 'minrect')


def _crop_polygon(page_img: Image.Image, polygon: list[tuple[int, int]]) -> Image.Image | None:
    if CROP_MODE == 'aabb':
        return _crop_polygon_aabb(page_img, polygon)
    try:
        return _crop_polygon_minrect(page_img, polygon)
    except Exception:
        return _crop_polygon_aabb(page_img, polygon)


def _local_tag(element: ET.Element) -> str:
    return element.tag.rsplit('}', 1)[-1]


def _parse_page_points(points: str | None) -> list[tuple[int, int]] | None:
    if not points:
        return None
    polygon = []
    for pair in points.split():
        x_str, _, y_str = pair.partition(',')
        if not y_str:
            return None
        polygon.append((int(round(float(x_str))), int(round(float(y_str)))))
    return polygon if len(polygon) >= 3 else None


def _parse_alto_polygon(points: str | None) -> list[tuple[int, int]] | None:
    """ALTO Shape/Polygon POINTS are space-separated 'x1 y1 x2 y2 ...' (no commas)."""
    if not points:
        return None
    values = points.replace(',', ' ').split()
    try:
        nums = [float(v) for v in values]
    except ValueError:
        return None
    polygon = [(int(round(nums[i])), int(round(nums[i + 1]))) for i in range(0, len(nums) - 1, 2)]
    return polygon if len(polygon) >= 3 else None


def _alto_bbox_polygon(textline: ET.Element) -> list[tuple[int, int]] | None:
    """Rectangle from the TextLine HPOS/VPOS/WIDTH/HEIGHT attributes, for ALTO without a Shape."""
    a = textline.attrib
    if not all(k in a for k in ('HPOS', 'VPOS', 'WIDTH', 'HEIGHT')):
        return None
    try:
        x, y, w, h = float(a['HPOS']), float(a['VPOS']), float(a['WIDTH']), float(a['HEIGHT'])
    except ValueError:
        return None
    if w < 1 or h < 1:
        return None
    return [(int(x), int(y)), (int(x + w), int(y)), (int(x + w), int(y + h)), (int(x), int(y + h))]


def extract_page_line_regions(
    xml_string: str
) -> tuple[str | None, list[tuple[str | None, list[tuple[int, int]], str]]]:
    """Returns the page image filename and per-TextLine (id, polygon, text) from PAGE or ALTO
    XML, namespace-agnostic. ALTO line text is its String CONTENT tokens joined by spaces."""
    root = ET.fromstring(xml_string)
    image_filename = None
    regions = []
    for element in root.iter():
        tag = _local_tag(element)
        if tag == 'Page' and image_filename is None:
            image_filename = element.get('imageFilename')          # PAGE
        elif tag == 'fileName' and image_filename is None and element.text:
            image_filename = element.text.strip()                  # ALTO sourceImageInformation
        elif tag == 'TextLine':
            polygon = None
            text = None
            strings = []
            for child in element.iter():
                child_tag = _local_tag(child)
                if child_tag == 'Coords' and polygon is None:
                    polygon = _parse_page_points(child.get('points'))      # PAGE
                elif child_tag == 'Polygon' and polygon is None:
                    polygon = _parse_alto_polygon(child.get('POINTS'))     # ALTO Shape/Polygon
                elif child_tag == 'Unicode' and text is None and child.text:
                    text = child.text.strip()                              # PAGE
                elif child_tag == 'String' and child.get('CONTENT') is not None:
                    strings.append(child.get('CONTENT'))                   # ALTO word/line token
            if text is None and strings:
                text = ' '.join(s for s in strings if s).strip()
            if polygon is None:
                polygon = _alto_bbox_polygon(element)                      # ALTO bbox fallback
            if polygon and text:
                regions.append((element.get('id') or element.get('ID'), polygon, text))
    return image_filename, regions


def _is_junk_path(path: Path) -> bool:
    """AppleDouble / macOS archive cruft that ships inside many Zenodo zips."""
    return path.name.startswith('._') or any(part == '__MACOSX' for part in path.parts)


def _build_image_stem_index(root_path: Path, image_extensions: list[str]) -> dict[str, Path]:
    """Maps image stem -> path across the whole source tree, so pages whose scans live in a
    sibling directory with an arbitrary name (not just 'images/') still resolve."""
    exts = {e.lower() for e in image_extensions}
    index: dict[str, Path] = {}
    for path in root_path.rglob('*'):
        if path.suffix.lower() in exts and path.is_file() and not _is_junk_path(path):
            index.setdefault(path.stem, path)
    return index


def _resolve_page_image(
    xml_path: Path, image_filename: str | None, image_extensions: list[str],
    stem_index: dict[str, Path] | None = None
) -> Path | None:
    if image_filename:
        candidate = xml_path.parent / image_filename
        if candidate.exists():
            return candidate

    stem = xml_path.stem
    search_dirs = [xml_path.parent, xml_path.parent.parent / 'images', xml_path.parent / 'images']
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for ext in image_extensions:
            candidate = search_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate

    if stem_index is not None:   # last resort: match by stem anywhere under the source root
        hit = stem_index.get(stem)
        if hit is None and image_filename:
            hit = stem_index.get(Path(image_filename).stem)
        if hit is not None:
            return hit
    return None


_CROP_CTX: dict = {}


def _crop_worker_init(root_path, out_root, dataset_name, image_extensions, stem_index):
    """Stashes the shared context once per worker so the stem index is not re-pickled per task,
    and lifts PIL's decompression-bomb cap for large scans."""
    Image.MAX_IMAGE_PIXELS = None
    _CROP_CTX.update(root_path=root_path, out_root=out_root, dataset_name=dataset_name,
                     image_extensions=image_extensions, stem_index=stem_index)


def _crop_page_task(xml_path: Path) -> tuple[int, str]:
    """Returns (n_cropped, status)."""
    ctx = _CROP_CTX
    try:
        image_filename, regions = extract_page_line_regions(xml_path.read_text(encoding='utf-8'))
    except Exception:
        return 0, 'unparseable'
    if not regions:
        return 0, 'no_regions'
    page_path = _resolve_page_image(xml_path, image_filename, ctx['image_extensions'], ctx['stem_index'])
    if page_path is None:
        return 0, 'no_image'
    try:
        page_img = Image.open(page_path)
        page_img.load()
    except Exception:
        return 0, 'unreadable'

    dest_dir = ctx['out_root'] / ctx['dataset_name'] / xml_path.parent.relative_to(ctx['root_path'])
    dest_dir.mkdir(parents=True, exist_ok=True)
    page_stem = xml_path.stem
    n = 0
    for idx, (line_id, polygon, text) in enumerate(regions):
        crop = _crop_polygon(page_img, polygon)
        if crop is None or crop.size[1] < MIN_LINE_HEIGHT:
            continue
        line_name = f"{page_stem}_{line_id or f'line{idx:04d}'}"
        crop.save(dest_dir / f"{line_name}.png")
        (dest_dir / f"{line_name}.gt.txt").write_text(text, encoding='utf-8')
        n += 1
    return n, 'ok'


def process_page_xml_dataset(root_dir: str | Path, output_dir: Path, image_extensions: list[str] | None = None) -> int:
    """Crops each labeled TextLine of a PAGE/ALTO dataset into line image + text pairs, using
    OCR_CROP_WORKERS processes (default 32)."""
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']

    root_path = Path(root_dir)
    dataset_name = root_path.name
    xml_paths = sorted(
        path for path in root_path.rglob('*.xml')
        if not any(part.startswith('_') for part in path.relative_to(root_path).parts)
        and not _is_junk_path(path)
    )

    # Crops are deterministic; without this every prepare run re-cropped ~127k identical lines.
    existing = output_dir / dataset_name
    if existing.is_dir():
        done = sum(1 for _ in existing.rglob('*.gt.txt'))
        if done:
            print(f"Reusing {done} existing line crops for {dataset_name} "
                  f"(delete {existing} to force a re-crop)")
            return done

    stem_index = _build_image_stem_index(root_path, image_extensions)
    n_workers = int(os.environ.get('OCR_CROP_WORKERS', min(32, (os.cpu_count() or 8))))
    print(f"Cropping line regions from {len(xml_paths)} PAGE/ALTO XML files in {dataset_name} "
          f"({n_workers} workers)...", flush=True)
    count = 0
    status: Counter = Counter()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_crop_worker_init,
                             initargs=(root_path, output_dir, dataset_name, image_extensions, stem_index)) as ex:
        for n, st in ex.map(_crop_page_task, xml_paths, chunksize=16):
            count += n
            status[st] += 1

    if status['no_image']:
        print(f"  {status['no_image']} pages had no resolvable image (skipped, e.g. transcription-only pages)")
    if status['unparseable'] or status['unreadable']:
        print(f"  {status['unparseable']} unparseable XML, {status['unreadable']} unreadable images")
    print(f"Extracted {count} line crops from {dataset_name}", flush=True)
    return count


def is_page_xml_dataset(root_path: Path, sample_size: int = 80, image_extensions: list[str] | None = None) -> bool:
    """True on the first sampled page that yields line regions and a resolvable scan."""
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']

    stem_index = _build_image_stem_index(root_path, image_extensions)
    img_stems = set(stem_index)

    def _skip(xml_path: Path) -> bool:
        return _is_junk_path(xml_path) or any(p.startswith('_') for p in xml_path.relative_to(root_path).parts)

    def _yields(xml_path: Path) -> bool:
        try:
            image_filename, regions = extract_page_line_regions(xml_path.read_text(encoding='utf-8'))
        except Exception:
            return False
        return bool(regions) and _resolve_page_image(xml_path, image_filename, image_extensions, stem_index) is not None

    # The Dutch VOC record has ~1M transcription-only pages around a few thousand with scans, so a
    # blind sample almost never hits one; try the image-bearing stems first.
    checked = 0
    for xml_path in root_path.rglob('*.xml'):
        if _skip(xml_path) or xml_path.stem not in img_stems:
            continue
        if _yields(xml_path):
            return True
        checked += 1
        if checked >= sample_size:
            break

    # Some datasets name their scan via imageFilename, unrelated to the xml stem.
    checked = 0
    for xml_path in root_path.rglob('*.xml'):
        if _skip(xml_path):
            continue
        if _yields(xml_path):
            return True
        checked += 1
        if checked >= sample_size:
            break
    return False


def download_from_label_studio(
    url: str, api_key: str, project_id: int, output_dir: Path, export_type: str = 'JSON'
) -> int:
    """Exports annotated tasks from a Label Studio project and crops each labeled region to a line pair."""
    print(f"Connecting to Label Studio at {url} (Project: {project_id})...")
    ls = Client(url=url, api_key=api_key)
    project = ls.get_project(project_id)

    export_result = project.export_tasks(export_type=export_type)
    if isinstance(export_result, Path):
        with open(export_result, 'r') as f:
            tasks = json.load(f)
    else:
        tasks = export_result

    ls_data_dir = output_dir / f"ls_project_{project_id}"
    ls_data_dir.mkdir(parents=True, exist_ok=True)
    page_cache_dir = ls_data_dir / "_pages"
    page_cache_dir.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": f"Token {api_key}"}
    count = 0

    for task in tasks:
        annotations = task.get('annotations') or []
        if not annotations:
            continue

        image_url = task['data'].get('image') or task['data'].get('ocr')
        if not image_url:
            continue

        page_filename = Path(image_url.split('?')[0]).name
        page_stem = Path(page_filename).stem
        page_path = page_cache_dir / page_filename

        if not page_path.exists():
            full_url = image_url if image_url.startswith('http') else f"{url.rstrip('/')}{image_url}"
            try:
                r = requests.get(full_url, headers=headers, stream=True)
                if r.status_code != 200:
                    print(f"Failed to download image {full_url}: Status {r.status_code}")
                    continue
                with open(page_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            except Exception as e:
                print(f"Error downloading {full_url}: {e}")
                continue

        try:
            page_img = Image.open(page_path)
            page_img.load()
        except Exception as e:
            print(f"Skipping unreadable page {page_path}: {e}")
            continue

        results = annotations[0].get('result', [])
        regions: dict[str, dict] = {}
        for res in results:
            rid = res.get('id')
            if not rid:
                continue
            entry = regions.setdefault(rid, {})
            res_type = res.get('type')
            value = res.get('value', {})
            page_w = res.get('original_width') or page_img.size[0]
            page_h = res.get('original_height') or page_img.size[1]
            if res_type in ('rectanglelabels', 'rectangle'):
                entry['polygon'] = _rect_to_polygon(value, page_w, page_h)
            elif res_type in ('polygonlabels', 'polygon'):
                entry['polygon'] = _poly_to_polygon(value, page_w, page_h)
            elif res_type == 'textarea':
                texts = value.get('text') or []
                entry['text'] = " ".join(texts).strip()

        for rid, entry in regions.items():
            polygon = entry.get('polygon')
            text = entry.get('text')
            if not polygon or not text:
                continue
            crop = _crop_polygon(page_img, polygon)
            if crop is None or crop.size[1] < MIN_LINE_HEIGHT:
                continue
            line_name = f"ls_project_{project_id}_{page_stem}_{rid}"
            crop.save(ls_data_dir / f"{line_name}.png")
            (ls_data_dir / f"{line_name}.gt.txt").write_text(text, encoding='utf-8')
            count += 1

    print(f"Successfully synced {count} line crops from Label Studio to {ls_data_dir}")
    return count


def generate_unique_filename(image_path: Path, folder: str, index: int) -> str:
    content = f"{image_path}_{folder}_{index}"
    hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]

    folder_clean = folder.replace('/', '_').replace('\\', '_').replace('.', '_') or 'root'
    return f"{folder_clean}_{index:06d}_{hash_suffix}"


def find_image_text_pairs(root_dir: str | Path,
                          image_extensions: list[str] | None = None) -> list[tuple[Path, Path, str]]:
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']

    ext_set = set(ext.lower() for ext in image_extensions)
    text_extensions = ['.gt.txt', '.corrected.txt', '.txt', '.xml']

    root_path = Path(root_dir)
    text_files_map = defaultdict(list)
    image_files = []

    for path in root_path.rglob('*'):
        if not path.is_file():
            continue

        if any(part.startswith('_') for part in path.relative_to(root_path).parts):
            continue

        lower_name = path.name.lower()
        if path.suffix.lower() in ext_set:
            image_files.append(path)
        else:
            for text_ext in text_extensions:
                if lower_name.endswith(text_ext):
                    base_name = path.name[:-len(text_ext)]
                    text_files_map[base_name].append(path)

    pairs = []

    # An image can have several ground-truth variants (.gt.txt, .corrected.txt); one is chosen below.
    for img_path in image_files:
        candidates = text_files_map.get(img_path.stem)

        if not candidates:
            continue

        img_parts = img_path.parent.parts
        shared_by_candidate = {}
        for cand in candidates:
            shared = 0
            for p1, p2 in zip(img_parts, cand.parent.parts):
                if p1 == p2:
                    shared += 1
                else:
                    break
            shared_by_candidate[cand] = shared

        # Same-stem images in different directories: keep the text files nearest this one.
        max_shared = max(shared_by_candidate.values())
        relative_folder = str(img_path.relative_to(root_path).parent)
        nearest = [c for c in sorted(candidates, key=lambda c: c.name)
                   if shared_by_candidate[c] == max_shared]
        chosen = _select_text_variant(nearest)
        if chosen is not None:
            pairs.append((img_path, chosen, relative_folder))

    return pairs


# Catches only a correction that is prose, truncated or empty, not hallucination: a fabricated
# digit ('20.' -> '120.') scores 0.050, real fixes 0.054-0.077, so no lower threshold separates them.
CORRECTION_MAX_EDIT_RATIO = float(os.environ.get('PREP_CORRECTION_MAX_EDIT', 0.60))


def _edit_ratio(a: str, b: str) -> float:
    """Levenshtein distance normalised by the longer string. 0.0 identical, 1.0 disjoint."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def _select_text_variant(candidates: list[Path]) -> Path | None:
    """Picks one ground truth per image: .gt.txt, then .corrected.txt, then .txt.

    Emitting every variant gave the same pixels conflicting targets (41.9% of one corpus's .txt and
    .corrected.txt disagree). .corrected.txt is a post-corrected transcription; checked
    against the images, it keeps period spelling the raw transcriber modernised (hauendo, not
    havendo), so it wins unless it fails the CORRECTION_MAX_EDIT_RATIO guard."""
    by_ext = {}
    for cand in candidates:
        for ext in ('.gt.txt', '.corrected.txt', '.txt', '.xml'):
            if cand.name.lower().endswith(ext):
                by_ext.setdefault(ext, cand)
                break

    corrected, raw = by_ext.get('.corrected.txt'), by_ext.get('.txt')
    if corrected is not None and raw is not None:
        try:
            c = corrected.read_text(encoding='utf-8').strip()
            r = raw.read_text(encoding='utf-8').strip()
        except OSError:
            return corrected
        if r and _edit_ratio(c, r) > CORRECTION_MAX_EDIT_RATIO:
            return raw

    for ext in ('.gt.txt', '.corrected.txt', '.txt', '.xml'):
        if ext in by_ext:
            return by_ext[ext]
    return None


def extract_xml_textlines(xml_string: str) -> list[str]:
    root = ET.fromstring(xml_string)
    namespace = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    return [
        u.text
        for line in root.findall('.//page:TextLine', namespace)
        if (u := line.find('.//page:Unicode', namespace)) is not None and u.text
    ]


def filter_single_line_pairs(pairs: list[tuple[Path, Path, str]]) -> list[tuple[Path, Path, str]]:
    def check_single_line(pair):
        _, text_path, _ = pair
        try:
            if text_path.suffix == '.xml':
                text_lines = extract_xml_textlines(text_path.read_text(encoding='utf-8'))
                return pair if len(text_lines) == 1 else None

            content = text_path.read_text(encoding='utf-8').rstrip('\n')
            return pair if '\n' not in content else None
        except Exception as e:
            print(f"Error reading {text_path}: {e}")
            return None

    with ThreadPoolExecutor() as executor:
        return [res for res in executor.map(check_single_line, pairs) if res]


def _hash_image_bytes(image_path: Path) -> str | None:
    try:
        hasher = hashlib.blake2b(digest_size=16)
        with open(image_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError as e:
        print(f"Error hashing {image_path}: {e}")
        return None


def deduplicate_pairs_by_image_hash(pairs: list[tuple[Path, Path, str]]) -> list[tuple[Path, Path, str]]:
    """Drops pairs whose image bytes duplicate an earlier image, keeping the first occurrence."""
    unique_paths = list(dict.fromkeys(pair[0] for pair in pairs))
    # blake2b plus the file read is GIL-bound; a thread pool pinned to one core.
    n_workers = int(os.environ.get('OCR_HASH_WORKERS', min(32, (os.cpu_count() or 8))))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        hashes = executor.map(_hash_image_bytes, unique_paths, chunksize=256)
        hash_by_path = dict(zip(unique_paths, hashes))

    kept_path_by_hash: dict[str, Path] = {}
    for path in unique_paths:
        image_hash = hash_by_path[path]
        if image_hash is not None:
            kept_path_by_hash.setdefault(image_hash, path)

    deduped = []
    for pair in pairs:
        image_hash = hash_by_path[pair[0]]
        # unhashable images are kept rather than silently dropped
        if image_hash is None or kept_path_by_hash[image_hash] == pair[0]:
            deduped.append(pair)

    dropped = len(pairs) - len(deduped)
    if dropped:
        print(f"Dropped {dropped} pairs whose image content duplicates an earlier pair")
    return deduped


def _pair_matches_sources(pair: tuple[Path, Path, str], sources: list[str]) -> bool:
    path_str = str(pair[0])
    for src in sources:
        stripped = src.strip('/')
        if stripped and stripped in path_str:
            return True
        # PAGE XML crops live under data_sources/_page_lines/<leaf>/, so match the leaf too.
        leaf = stripped.rsplit('/', 1)[-1]
        if leaf and f"/{leaf}/" in path_str:
            return True
    return False


def stratified_split(
    pairs: list[tuple[Path, Path, str]],
    train_ratio: float = 0.9,
    seed: int = 42,
    valid_sources: list[str] | None = None
) -> tuple[list, list]:
    """Splits pairs into train and valid per folder, keeping every crop of an image together.

    When valid_sources is given, only pairs whose source path contains one of those substrings
    are eligible for the validation split; every other pair goes entirely to training."""
    random.seed(seed)

    if valid_sources:
        eligible, train_only = [], []
        for pair in pairs:
            (eligible if _pair_matches_sources(pair, valid_sources) else train_only).append(pair)
        print(
            f"Validation restricted to {valid_sources}: "
            f"{len(eligible)} eligible pairs, {len(train_only)} train-only pairs"
        )
        if not eligible:
            print("WARNING: no pairs matched valid_sources; validation set will be empty.")
    else:
        eligible, train_only = pairs, []

    folder_groups = defaultdict(list)
    for pair in eligible:
        folder_groups[pair[2]].append(pair)

    train_pairs, valid_pairs = [], []
    for folder_pairs in folder_groups.values():
        image_groups = defaultdict(list)
        for pair in folder_pairs:
            image_groups[pair[0]].append(pair)
        grouped = list(image_groups.values())
        random.shuffle(grouped)
        split_idx = int(len(grouped) * train_ratio)
        for group in grouped[:split_idx]:
            train_pairs.extend(group)
        for group in grouped[split_idx:]:
            valid_pairs.extend(group)

    train_pairs.extend(train_only)

    random.shuffle(train_pairs)
    random.shuffle(valid_pairs)
    return train_pairs, valid_pairs


def load_existing_manifest(output_dir: Path) -> dict[str, dict[str, str]]:
    manifest_path = output_dir / 'split_manifest.json'
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return {'train': {}, 'valid': {}}


def save_manifest(output_dir: Path, train_mapping: dict[str, str], valid_mapping: dict[str, str]):
    manifest_path = output_dir / 'split_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump({'train': train_mapping, 'valid': valid_mapping}, f, indent=2)


def _text_variant_suffix(text_path: Path) -> str:
    if text_path.name.endswith('.corrected.txt'):
        return '.corrected.txt'
    if text_path.name.endswith('.gt.txt'):
        return '.gt.txt'
    return '.txt'


def copy_file_with_rename(args) -> tuple[bool, str | None]:
    """Writes the canonicalised text under unique_name, keeping its variant suffix so training can
    filter on it; the image is placed only for the first text of each unique_name."""
    image_path, text_path, dest_dir, unique_name, place_image = args
    try:
        new_txt_path = dest_dir / f"{unique_name}{_text_variant_suffix(text_path)}"

        if text_path.suffix == '.xml':
            text_lines = extract_xml_textlines(text_path.read_text(encoding='utf-8'))
            text = text_lines[0]
        else:
            text = text_path.read_text(encoding='utf-8')
        new_txt_path.write_text(canonicalize(text), encoding='utf-8')

        if place_image:
            new_img_path = dest_dir / f"{unique_name}{image_path.suffix}"
            _place_image(image_path, new_img_path, IMAGE_LINK_MODE)
        return True, unique_name
    except Exception as e:
        print(f"Error copying {image_path}: {e}")
        return False, None


def split_and_save_pairs(
    pairs: list[tuple[Path, Path, str]],
    output_dir: str,
    train_ratio: float = 0.9,
    seed: int = 42,
    mode: Literal['keep', 'extend', 'reset'] = 'reset',
    num_workers: int | None = None,
    valid_sources: list[str] | None = None
):
    output_path = Path(output_dir)
    train_dir = output_path / 'train'
    valid_dir = output_path / 'valid'
    num_workers = num_workers or os.cpu_count()

    manifest = load_existing_manifest(output_path) if mode != 'reset' else {'train': {}, 'valid': {}}
    train_mapping = manifest.get('train', {})
    valid_mapping = manifest.get('valid', {})

    existing_train_paths = set(train_mapping.values())
    existing_valid_paths = set(valid_mapping.values())
    all_existing_paths = existing_train_paths | existing_valid_paths

    if mode == 'reset':
        print("Resetting dataset...")
        shutil.rmtree(train_dir, ignore_errors=True)
        shutil.rmtree(valid_dir, ignore_errors=True)
        manifest_path = output_path / 'split_manifest.json'
        if manifest_path.exists():
            manifest_path.unlink()
        train_mapping = {}
        valid_mapping = {}
        train_pairs, valid_pairs = stratified_split(pairs, train_ratio, seed, valid_sources)

    elif mode == 'keep':
        print("Keeping existing validation set...")
        new_pairs = [p for p in pairs if str(p[0]) not in all_existing_paths]
        train_pairs, _ = stratified_split(new_pairs, 1.0, seed, valid_sources)
        valid_pairs = []

    elif mode == 'extend':
        print("Extending existing dataset...")
        new_pairs = [p for p in pairs if str(p[0]) not in all_existing_paths]
        train_pairs, valid_pairs = stratified_split(new_pairs, train_ratio, seed, valid_sources)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    starting_index = len(train_mapping) + len(valid_mapping)

    def process_split(split_pairs, dest_dir, mapping):
        args_list = []
        name_by_image = {}
        next_index = starting_index + len(mapping)
        for img, txt, folder in split_pairs:
            unique_name = name_by_image.get(img)
            place_image = unique_name is None
            if place_image:
                unique_name = generate_unique_filename(img, folder, next_index)
                name_by_image[img] = unique_name
                mapping[unique_name] = str(img)
                next_index += 1
            args_list.append((img, txt, dest_dir, unique_name, place_image))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(copy_file_with_rename, args_list))

    if train_pairs:
        print(f"Copying {len(train_pairs)} pairs to train...")
        process_split(train_pairs, train_dir, train_mapping)

    if valid_pairs:
        print(f"Copying {len(valid_pairs)} pairs to valid...")
        process_split(valid_pairs, valid_dir, valid_mapping)

    save_manifest(output_path, train_mapping, valid_mapping)

    print(f"\n{'='*50}\nSplit complete!")
    print(f"Train: {len(train_mapping)} samples | Valid: {len(valid_mapping)} samples")


def _limit_pairs(pairs: list, limit: int | None, seed: int) -> list:
    if limit is None or len(pairs) <= limit:
        return pairs
    return random.Random(seed).sample(pairs, limit)


def process_ocr_dataset(
    root_dirs: list[str | dict],
    output_dir: str,
    train_ratio: float = 0.9,
    mode: Literal['keep', 'extend', 'reset'] = 'reset',
    seed: int = 42,
    ls_configs: Optional[list[Dict]] = None,
    valid_sources: Optional[list[str]] = None,
    deduplicate: bool = True
):
    """Each root_dirs entry is either a path string or a dict {"path": ..., "limit": int | None};
    limit caps how many pairs are randomly drawn from that source (None uses all)."""
    sources = []
    for entry in root_dirs:
        path, limit = (entry['path'], entry.get('limit')) if isinstance(entry, dict) else (entry, None)
        path = Path(path)
        if path.exists():
            sources.append((path, limit))
        else:
            print(f"Skipping missing source: {path}")

    label_studio_root = Path('data_sources/label_studio')
    label_studio_root.mkdir(parents=True, exist_ok=True)

    if ls_configs:
        for ls_config in ls_configs:
            download_from_label_studio(
                url=ls_config['url'],
                api_key=ls_config['api_key'],
                project_id=ls_config['project_id'],
                output_dir=label_studio_root
            )
        sources.append((label_studio_root, None))

    print(f"Searching for image-text pairs in {len(sources)} directories...")
    pairs = []
    page_lines_root = Path('data_sources/_page_lines')
    for root_path, limit in sources:
        if is_page_xml_dataset(root_path):
            print(f"Detected multi-line PAGE XML dataset: {root_path.name}")
            page_lines_root.mkdir(parents=True, exist_ok=True)
            process_page_xml_dataset(root_path, page_lines_root)
            src_pairs = find_image_text_pairs(page_lines_root / root_path.name)
        else:
            src_pairs = find_image_text_pairs(root_path)

        limited = _limit_pairs(src_pairs, limit, seed)
        if len(limited) != len(src_pairs):
            print(f"  {root_path}: using {len(limited)} of {len(src_pairs)} pairs (limit={limit})")
        else:
            print(f"  {root_path}: {len(src_pairs)} pairs")
        pairs.extend(limited)

    print(f"Found {len(pairs)} pairs")

    print("\nFiltering for single-line ground truth...")
    filtered_pairs = filter_single_line_pairs(pairs)
    print(f"Kept {len(filtered_pairs)} single-line pairs")

    if deduplicate:
        print("\nRemoving duplicate images by content hash...")
        filtered_pairs = deduplicate_pairs_by_image_hash(filtered_pairs)
    else:
        print("\nSkipping duplicate-image hashing")

    if filtered_pairs:
        split_and_save_pairs(filtered_pairs, output_dir, train_ratio, seed, mode, valid_sources=valid_sources)
    else:
        print("No pairs to process!")


if __name__ == "__main__":
    LS_URL = os.getenv("LABEL_STUDIO_URL", "")
    LS_API_KEY = os.getenv("LABEL_STUDIO_LEGACY_API_KEY", "")
    LABEL_STUDIO_PROJECT_IDS: list[int] = []

    print(f"Using Label Studio URL: {LS_URL}")

    label_studio_projects = [
        {"url": LS_URL, "api_key": LS_API_KEY, "project_id": project_id}
        for project_id in LABEL_STUDIO_PROJECT_IDS
    ]

    # A short example of corpora under data_sources/.
    process_ocr_dataset(
        root_dirs=[
            {"path": "data_sources/catmus_medieval", "limit": None},
            {"path": "data_sources/catmus_modern/pairs", "limit": None},
            {"path": "data_sources/gt4histocr/corpus/dta19", "limit": None},
            {"path": "data_sources/NotarialDeeds_HTR", "limit": None},
            {"path": "data_sources/reichsanzeiger", "limit": None},
            {"path": "data_sources/party/HTR-United__cremma-medieval", "limit": 50000},
        ],
        output_dir="data",
        mode='reset',
        train_ratio=0.90,
        ls_configs=label_studio_projects,
        valid_sources=None,
        # PREP_DEDUPE=0 skips hashing every image when the sources have not changed since the last run
        deduplicate=os.getenv("PREP_DEDUPE", "1") != "0",
    )
