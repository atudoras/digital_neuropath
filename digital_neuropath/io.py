"""Read one field of view (fov) from a Leica .lif series or an OME or ImageJ TIFF, only the channels asked for.
Chain: list_series -> read_fov -> Fov(data, spacing_um, ...), plus read_rows / write_rows for sheets.
A channel that is never read cannot bias a call, so the blind is what is in memory, not what is shown.
"""
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

__all__ = ["Fov", "SeriesInfo", "append_row", "fov_id", "list_series", "read_fov", "read_rows", "write_rows"]

UM_PER_UNIT = {"µm": 1.0, "um": 1.0, "micron": 1.0, "nm": 1e-3, "mm": 1e3}
# A file whose allocated blocks hold less than this fraction of its size is a cloud placeholder.
ON_DISK_FRACTION = 0.99


@dataclass(frozen=True)
class SeriesInfo:
    """What one series holds, read from metadata without touching pixels."""
    source: str
    series: int
    name: str
    n_channels: int
    shape_zyx: tuple[int, int, int]
    spacing_um: tuple[float, float, float]
    channel_names: tuple[str, ...] = ()  # empty when the metadata names no channels


@dataclass(frozen=True)
class Fov:
    """The channels one step is allowed to hold, with the voxel size read from the file."""
    data: dict[str, np.ndarray]
    channel_index: dict[str, int]
    spacing_um: tuple[float, float, float]
    source: str
    series: int
    name: str

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape (z, y, x) shared by every loaded channel."""
        z, y, x = next(iter(self.data.values())).shape
        return int(z), int(y), int(x)

    @property
    def fov_id(self) -> str:
        """Stable folder name for this fov: file stem plus series."""
        return fov_id(self.source, self.series)


def fov_id(source: str | Path, series: int) -> str:
    """Folder name for one series of one file."""
    return f"{Path(source).stem.removesuffix('.ome')}_s{series:02d}"


def list_series(path: str | Path, series: list[int] | int | None = None) -> list[SeriesInfo]:
    """Every series in a .lif (or only those listed in `series`), or the single series of an
    OME-TIFF, from metadata only. Only the series described are checked for a z scale, so a 2D
    overview elsewhere in the file does not block the stacks asked for."""
    path = Path(path)
    if isinstance(series, int):
        series = [series]
    check_on_disk(path)
    if path.suffix.lower() == ".lif":
        from readlif.reader import LifFile

        lif = LifFile(str(path))
        unknown = sorted(set(series or []) - set(range(lif.num_images)))
        if unknown:
            raise ValueError(f"{path.name} has series 0-{lif.num_images - 1}; {unknown} do not exist")
        wanted = range(lif.num_images) if series is None else sorted(set(series))
        return [lif_info(path, i, lif.get_image(i)) for i in wanted]
    return [tiff_info(path)]


def check_on_disk(path: Path) -> None:
    """Refuse a file whose bytes are not on this disk (a Box/OneDrive placeholder), which would
    hang or download mid-read."""
    info = path.stat()
    if info.st_blocks * 512 < ON_DISK_FRACTION * info.st_size:
        raise OSError(f"{path.name}: cloud placeholder (Box/OneDrive): make it available offline "
                      f"({info.st_blocks * 512} of {info.st_size} bytes on disk)")


def lif_info(path: Path, series: int, image) -> SeriesInfo:
    """SeriesInfo for one readlif image; refuses a series with no z step in its metadata."""
    scale = image.scale_n
    if not all(scale.get(k) for k in (1, 2, 3)):
        raise ValueError(f"{path.name} series {series} ({image.name}) has no x/y/z scale in its "
                         f"metadata ({scale}); voxel size is read from the file, never typed.")
    spacing = (1.0 / float(scale[3]), 1.0 / float(scale[2]), 1.0 / float(scale[1]))
    shape = (int(image.dims.z), int(image.dims.y), int(image.dims.x))
    return SeriesInfo(str(path), series, str(image.name), int(image.channels), shape, spacing)


def tiff_axes(handle) -> tuple[str, tuple[int, ...]]:
    """Axes string and shape of the first series, with singleton T dropped."""
    series = handle.series[0]
    axes, shape = series.axes, tuple(series.shape)
    keep = [i for i, a in enumerate(axes) if not (a in "TS" and shape[i] == 1)]
    axes = "".join(axes[i] for i in keep)
    shape = tuple(shape[i] for i in keep)
    if set(axes) - set("CZYX"):
        raise ValueError(f"unsupported OME-TIFF axes {series.axes!r}; expected C, Z, Y, X")
    return axes, shape


def ome_spacing(handle, path: Path) -> tuple[float, float, float]:
    """(z, y, x) voxel size in um from the OME-XML PhysicalSize attributes."""
    if not handle.ome_metadata:
        raise ValueError(f"{path.name} carries no OME metadata; voxel size is read from the file, "
                         f"never typed. Export it as OME-TIFF with physical sizes.")
    root = ElementTree.fromstring(handle.ome_metadata)
    pixels = next(el for el in root.iter() if el.tag.endswith("Pixels"))
    out = []
    for axis in "ZYX":
        value = pixels.get(f"PhysicalSize{axis}")
        unit = pixels.get(f"PhysicalSize{axis}Unit", "µm")
        if value is None or unit not in UM_PER_UNIT:
            raise ValueError(f"{path.name} has no usable PhysicalSize{axis} ({value!r} {unit!r}) "
                             f"in its OME metadata; voxel size is read from the file, never typed.")
        out.append(float(value) * UM_PER_UNIT[unit])
    return out[0], out[1], out[2]


def imagej_spacing(handle, path: Path) -> tuple[float, float, float]:
    """(z, y, x) voxel size in um from an ImageJ TIFF: z from its spacing, y and x from the
    resolution tags, all in its unit."""
    meta = handle.imagej_metadata or {}
    unit = str(meta.get("unit", "")).replace("\\u00B5", "µ")
    page = handle.pages[0]
    try:
        y_num, y_den = page.tags["YResolution"].value
        x_num, x_den = page.tags["XResolution"].value
    except KeyError:
        x_num = x_den = y_num = y_den = 1
    if unit not in UM_PER_UNIT or "spacing" not in meta or (x_num, x_den) == (1, 1):
        raise ValueError(f"{path.name} carries no voxel size (OME PhysicalSize, or ImageJ unit, "
                         f"spacing and resolution); voxel size is read from the file, never typed. "
                         f"Export it as OME-TIFF with physical sizes, or set the scale in ImageJ.")
    k = UM_PER_UNIT[unit]
    return float(meta["spacing"]) * k, y_den / y_num * k, x_den / x_num * k


def tiff_spacing(handle, path: Path) -> tuple[float, float, float]:
    """(z, y, x) voxel size in um from an OME-TIFF or an ImageJ TIFF."""
    return ome_spacing(handle, path) if handle.ome_metadata else imagej_spacing(handle, path)


def ome_channel_names(handle) -> tuple[str, ...]:
    """Channel names from the OME-XML, or () if any channel is unnamed."""
    if not handle.ome_metadata:
        return ()
    root = ElementTree.fromstring(handle.ome_metadata)
    pixels = next(el for el in root.iter() if el.tag.endswith("Pixels"))
    names = [el.get("Name") for el in pixels if el.tag.endswith("Channel")]
    return tuple(n for n in names if n) if names and all(names) else ()


def tiff_info(path: Path) -> SeriesInfo:
    """SeriesInfo for an OME or ImageJ TIFF, which holds one fov."""
    import tifffile

    with tifffile.TiffFile(path) as handle:
        axes, shape = tiff_axes(handle)
        spacing = tiff_spacing(handle, path)
        names = ome_channel_names(handle)
    size = dict(zip(axes, shape))
    return SeriesInfo(str(path), 0, path.stem, size.get("C", 1),
                      (size.get("Z", 1), size["Y"], size["X"]), spacing, names)


def read_fov(path: str | Path, series: int, channels: dict[str, int]) -> Fov:
    """Read only `channels` (name -> index) of one series; the others are never requested."""
    path = Path(path)
    check_on_disk(path)
    if path.suffix.lower() == ".lif":
        return read_lif(path, series, channels)
    if series != 0:
        raise ValueError(f"{path.name} is a single-fov OME-TIFF; series must be 0, got {series}")
    return read_tiff(path, channels)


def check_channels(channels: dict[str, int], n_channels: int, what: str) -> None:
    """Refuse a channel index the file does not have."""
    bad = {name: i for name, i in channels.items() if not 0 <= i < n_channels}
    if bad:
        raise ValueError(f"{what} has {n_channels} channels; requested indices {bad} do not exist. "
                         f"A channel map is declared, never guessed from pixels.")


def read_lif(path: Path, series: int, channels: dict[str, int]) -> Fov:
    """One .lif series, plane by plane, requested channels only."""
    from readlif.reader import LifFile

    image = LifFile(str(path)).get_image(series)
    info = lif_info(path, series, image)
    check_channels(channels, info.n_channels, f"{path.name} series {series}")
    data = {name: np.stack([np.asarray(image.get_frame(z=z, c=c))
                            for z in range(info.shape_zyx[0])])
            for name, c in channels.items()}
    return Fov(data, dict(channels), info.spacing_um, str(path), series, info.name)


def read_tiff(path: Path, channels: dict[str, int]) -> Fov:
    """An OME or ImageJ TIFF, reading only the pages that belong to the requested channels."""
    import tifffile

    with tifffile.TiffFile(path) as handle:
        axes, shape = tiff_axes(handle)
        spacing = tiff_spacing(handle, path)
        size = dict(zip(axes, shape))
        check_channels(channels, size.get("C", 1), path.name)
        page_axes = axes[:-2]
        page_shape = shape[:-2]
        data = {}
        for name, c in channels.items():
            index = []
            for z in range(size.get("Z", 1)):
                position = {"C": c, "Z": z}
                index.append(int(np.ravel_multi_index([position[a] for a in page_axes],
                                                      page_shape)) if page_axes else 0)
            planes = handle.series[0].asarray(key=index)
            data[name] = np.asarray(planes).reshape(size.get("Z", 1), size["Y"], size["X"])
    return Fov(data, dict(channels), spacing, str(path), 0, path.stem)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Rows of a sheet, skipping its `#` provenance header."""
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def write_rows(path: Path, fields: list[str], rows: list[dict], header: list[str]) -> None:
    """Write a sheet with its provenance as `#` lines above it, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        for line in header:
            handle.write(f"# {line}\n")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def append_row(path: Path, fields: list[str], row: dict, header: list[str]) -> None:
    """Append one row and fsync it, writing the header first if the file is new."""
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        if new:
            for line in header:
                handle.write(f"# {line}\n")
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
