from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

__version__ = "4.11.0-shim"

INTER_NEAREST = 0
INTER_LINEAR = 1
BORDER_CONSTANT = 0
COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 5
IMREAD_COLOR = 1
FONT_HERSHEY_SIMPLEX = 0
LINE_AA = 16


def setNumThreads(_count: int) -> None:
    return None


def cvtColor(image: np.ndarray, code: int) -> np.ndarray:
    if code in (COLOR_BGR2RGB, COLOR_RGB2BGR):
        return image[..., ::-1].copy()
    raise NotImplementedError(f"Unsupported color conversion code: {code}")


def resize(image: np.ndarray, dsize: tuple[int, int], interpolation: int = INTER_LINEAR) -> np.ndarray:
    resample = Image.NEAREST if interpolation == INTER_NEAREST else Image.BILINEAR
    pil_image = Image.fromarray(_to_uint8(image))
    return np.array(pil_image.resize(dsize, resample))


def copyMakeBorder(
    image: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    borderType: int = BORDER_CONSTANT,
    value=(0, 0, 0),
) -> np.ndarray:
    if borderType != BORDER_CONSTANT:
        raise NotImplementedError("Only BORDER_CONSTANT is supported by this shim")

    image = _to_uint8(image)
    if image.ndim == 2:
        fill_value = value[0] if isinstance(value, (tuple, list)) else value
        return np.pad(image, ((top, bottom), (left, right)), mode="constant", constant_values=fill_value)

    result = np.full(
        (image.shape[0] + top + bottom, image.shape[1] + left + right, image.shape[2]),
        _normalize_fill(value, image.shape[2]),
        dtype=image.dtype,
    )
    result[top : top + image.shape[0], left : left + image.shape[1]] = image
    return result


def imread(path: str, flags: int = IMREAD_COLOR) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    arr = np.array(image)
    if flags == IMREAD_COLOR:
        return arr[..., ::-1].copy()
    return arr


def imwrite(path: str, image: np.ndarray) -> bool:
    Image.fromarray(_to_uint8(image)).save(path)
    return True


def imdecode(buffer: np.ndarray, flags: int = IMREAD_COLOR) -> np.ndarray:
    image = Image.open(BytesIO(buffer.tobytes())).convert("RGB")
    arr = np.array(image)
    if flags == IMREAD_COLOR:
        return arr[..., ::-1].copy()
    return arr


def rectangle(
    image: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color,
    thickness: int = 1,
):
    pil_image = Image.fromarray(_to_uint8(image))
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle([pt1, pt2], outline=_color_tuple(color), width=max(thickness, 1))
    image[:] = np.array(pil_image)
    return image


def putText(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    fontFace: int = FONT_HERSHEY_SIMPLEX,
    fontScale: float = 1.0,
    color=(255, 255, 255),
    thickness: int = 1,
    lineType: int = LINE_AA,
):
    del fontFace, thickness, lineType
    pil_image = Image.fromarray(_to_uint8(image))
    draw = ImageDraw.Draw(pil_image)
    draw.text(org, text, fill=_color_tuple(color))
    image[:] = np.array(pil_image)
    return image


def getTextSize(
    text: str,
    fontFace: int = FONT_HERSHEY_SIMPLEX,
    fontScale: float = 1.0,
    thickness: int = 1,
):
    del fontFace, thickness
    width = max(int(len(text) * 10 * fontScale), 1)
    height = max(int(18 * fontScale), 1)
    return (width, height), 0


def imshow(*args, **kwargs):
    return None


def waitKey(delay: int = 0) -> int:
    del delay
    return -1


def destroyAllWindows() -> None:
    return None


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def _normalize_fill(value, channels: int) -> np.ndarray:
    if isinstance(value, (tuple, list)):
        items = list(value[:channels]) + [value[-1]] * max(channels - len(value), 0)
        return np.array(items[:channels], dtype=np.uint8)
    return np.full((channels,), value, dtype=np.uint8)


def _color_tuple(color) -> tuple[int, int, int]:
    if isinstance(color, (tuple, list)):
        vals = list(color[:3]) + [255] * max(3 - len(color), 0)
        return tuple(int(v) for v in vals[:3])
    return (int(color), int(color), int(color))
