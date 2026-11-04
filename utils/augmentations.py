import albumentations as A
import cv2
import numpy as np


def paper_colour(image: np.ndarray) -> np.ndarray:
    return np.percentile(image.reshape(-1, image.shape[-1]), 90, axis=0).astype(np.uint8)


class BaselineCurve(A.ImageOnlyTransform):
    """Bends the baseline into a gentle wave, as on a curved page or a drifting hand. The
    amplitude is a fraction of the line height."""

    def __init__(self, amplitude=(0.01, 0.04), waves=(0.5, 2.0), p=0.5):
        super().__init__(p=p)
        self.amplitude = amplitude
        self.waves = waves

    def get_params(self):
        return {'amplitude': self.py_random.uniform(*self.amplitude), 'waves': self.py_random.uniform(*self.waves),
                'phase': self.py_random.uniform(0, 2 * np.pi)}

    def apply(self, img, amplitude=0.0, waves=1.0, phase=0.0, **params):
        height, width = img.shape[:2]
        xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        shift = (amplitude * height * np.sin(2 * np.pi * waves * xs / width + phase)).astype(np.float32)
        return cv2.remap(img, xs, ys + shift, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class VerticalCropPad(A.ImageOnlyTransform):
    """A line finder's box never sits the same way twice: each edge is cut into (negative) or
    padded with paper (positive) by a fraction of the height, then the line is scaled back to its
    height so batches keep one height."""

    def __init__(self, limit=(-0.10, 0.15), p=0.5):
        super().__init__(p=p)
        self.limit = limit

    def get_params(self):
        return {'top': self.py_random.uniform(*self.limit), 'bottom': self.py_random.uniform(*self.limit)}

    def apply(self, img, top=0.0, bottom=0.0, **params):
        height, width = img.shape[:2]
        cut_top, cut_bottom = int(max(0.0, -top) * height), int(max(0.0, -bottom) * height)
        body = img[cut_top:height - cut_bottom]
        paper = paper_colour(img)
        pad_top = np.broadcast_to(paper, (int(max(0.0, top) * height), width, img.shape[2]))
        pad_bottom = np.broadcast_to(paper, (int(max(0.0, bottom) * height), width, img.shape[2]))
        framed = np.concatenate([pad_top, body, pad_bottom], axis=0)
        if framed.shape[0] == height:
            return np.ascontiguousarray(framed)
        return cv2.resize(framed, (width, height), interpolation=cv2.INTER_AREA)


class NeighbourLineBleed(A.ImageOnlyTransform):
    """Descenders of the line above (or ascenders of the line below) reaching into the crop. The
    intruding strip is cut from the lower or upper part of this line itself, shifted sideways,
    and laid over the opposite edge with a darken blend, so only ink is added."""

    def __init__(self, depth=(0.10, 0.25), p=0.5):
        super().__init__(p=p)
        self.depth = depth

    def get_params(self):
        return {'depth': self.py_random.uniform(*self.depth), 'from_above': self.py_random.random() < 0.5,
                'offset': self.py_random.uniform(0.0, 1.0)}

    def apply(self, img, depth=0.0, from_above=True, offset=0.0, **params):
        height, width = img.shape[:2]
        rows = max(1, int(depth * height))
        strip = img[height - rows:] if from_above else img[:rows]
        strip = np.roll(strip, int(offset * width), axis=1)
        out = img.copy()
        if from_above:
            out[:rows] = np.minimum(out[:rows], strip)
        else:
            out[height - rows:] = np.minimum(out[height - rows:], strip)
        return out


class Binarize(A.ImageOnlyTransform):
    """Black and white only, as in some older digitisations (Otsu threshold)."""

    def apply(self, img, **params):
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return np.repeat(binary[:, :, None], img.shape[2], axis=2)
