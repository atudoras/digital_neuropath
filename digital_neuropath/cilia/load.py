"""Reading a fov in, writing labels and tables out, and the declared shape of every table.
load -> read_plane, read_cmap, read_labels; write_labels; frame_of, write_table, read_table.
One writer and one reader per table, both checking the same schema, so a column cannot drift.
"""
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["Fov", "load", "SCHEMAS", "read_table", "write_table"]


@dataclass(frozen=True)
class Column:
    """One declared column: name, kind (int, float, bool, string), unit and nullability."""

    name: str
    kind: str
    unit: str = ""
    nullable: bool = False


DONOR = Column("donor", "string")
FOV = Column("fov", "string")

MORPHOMETRY = (
    Column("label", "int"), Column("n_voxels", "int", "voxels"),
    Column("volume_um3", "float", "µm³"), Column("centroid_z_um", "float", "µm"),
    Column("centroid_y_um", "float", "µm"), Column("centroid_x_um", "float", "µm"),
    Column("ac3_integrated", "float", "intensity"), Column("ac3_mean", "float", "intensity"),
)

SKELETON = (
    Column("label", "int"), Column("length_signal_um", "float", "µm"),
    Column("total_curvature_rad", "float", "rad", nullable=True),
    Column("total_curvature_deg", "float", "deg", nullable=True),
    Column("max_curvature_inv_um", "float", "1/µm", nullable=True),
    Column("axoneme_mean", "float", "intensity", nullable=True),
    Column("axoneme_max", "float", "intensity", nullable=True),
    Column("axoneme_cv", "float", "ratio", nullable=True),
    Column("axoneme_bg_median", "float", "intensity"),
    Column("axoneme_bg_sigma", "float", "intensity"), Column("branch_count", "int", "count"),
    Column("blob_r_eff_um", "float", "µm"), Column("blob_flag", "bool"),
)

PCNT_MORPHOMETRY = (
    Column("label", "int"), Column("n_voxels", "int", "voxels"),
    Column("volume_um3", "float", "µm³"), Column("centroid_z_um", "float", "µm"),
    Column("centroid_y_um", "float", "µm"), Column("centroid_x_um", "float", "µm"),
    Column("pcnt_integrated", "float", "intensity"), Column("pcnt_mean", "float", "intensity"),
    Column("surface_area_um2", "float", "µm²"), Column("sphericity", "float", "ratio"),
    Column("anisotropy", "float", "ratio"), Column("pcnt_bg_median", "float", "intensity"),
    Column("pcnt_bg_sigma", "float", "intensity"),
)

CENSORING = (
    DONOR, FOV, Column("neuron_uid", "string"),
    Column("neuron_id", "int"), Column("group", "string"), Column("n_voxels", "int", "voxels"),
    Column("volume_um3", "float", "µm³"), Column("touches_z_min", "bool"),
    Column("touches_z_max", "bool"), Column("touches_y_min", "bool"),
    Column("touches_y_max", "bool"), Column("touches_x_min", "bool"),
    Column("touches_x_max", "bool"), Column("z_span_planes", "int", "planes"),
    Column("z_margin_um", "float", "µm"), Column("lateral_margin_um", "float", "µm"),
    Column("censoring", "string"), Column("producer_rule_fires", "bool", nullable=True),
    Column("producer_excluded", "bool"),
)

GEOMETRY = (
    Column("label", "int"), Column("owner_neuron", "int"), Column("owner_by", "string"),
    Column("n_voxels", "int", "voxels"), Column("volume_um3", "float", "µm³"),
    Column("frac_in_soma", "float", "fraction"), Column("frac_in_nucleus", "float", "fraction"),
    Column("containment", "string"), Column("base_dist_um", "float", "µm", nullable=True),
    Column("tip_out_um", "float", "µm", nullable=True),
    Column("sdf_span_um", "float", "µm", nullable=True),
    Column("outward_cos", "float", "cosine", nullable=True), Column("aspect", "float", "ratio"),
    Column("centroid_z", "float", "voxels"), Column("centroid_y", "float", "voxels"),
    Column("centroid_x", "float", "voxels"), Column("base_z", "float", "voxels", nullable=True),
    Column("base_y", "float", "voxels", nullable=True),
    Column("base_x", "float", "voxels", nullable=True),
    Column("tip_z", "float", "voxels", nullable=True),
    Column("tip_y", "float", "voxels", nullable=True),
    Column("tip_x", "float", "voxels", nullable=True),
)

CILIA = (
    DONOR, FOV, Column("cilium_uid", "string"),
    Column("neuron_uid", "string"), *GEOMETRY, Column("ac3_mean", "float", "intensity"),
    Column("ac3_integrated", "float", "intensity"),
    Column("length_signal_um", "float", "µm", nullable=True),
    Column("blob_flag", "bool", nullable=True),
    Column("blob_r_eff_um", "float", "µm", nullable=True),
    Column("branch_count", "int", "count", nullable=True), Column("nominated", "bool"),
    Column("nomination_reason", "string"), Column("eligible", "bool"),
    Column("exclusion_reason", "string"), Column("chosen", "bool"),
)

NEURONS = (
    DONOR, FOV, Column("neuron_uid", "string"),
    Column("neuron_id", "int"), Column("soma_n_voxels", "int", "voxels"),
    Column("soma_volume_um3", "float", "µm³"), Column("pcnt_mean", "float", "intensity"),
    Column("pcnt_median", "float", "intensity"), Column("pcnt_p99", "float", "intensity"),
    Column("pcnt_max", "float", "intensity"), Column("pcnt_integrated", "float", "intensity"),
    Column("pcnt_integrated_bgsub", "float", "intensity"), Column("pcnt_gini", "float", "ratio"),
    Column("pcnt_top1pct_frac", "float", "fraction", nullable=True),
    Column("pcnt_p99_over_bg", "float", "ratio", nullable=True), Column("n_peaks", "int", "count"),
    Column("peak_density_per_100um3", "float", "1/100µm³"), Column("n_foci_inside", "int", "count"),
    Column("n_foci_in_territory", "int", "count"), Column("foci_volume_um3_total", "float", "µm³"),
    Column("foci_volume_um3_max", "float", "µm³"),
    Column("foci_integrated_total", "float", "intensity"),
    Column("foci_integrated_max", "float", "intensity"), Column("cilium_uid", "string"),
    Column("has_cilium", "bool"), Column("cilium_tip_out_um", "float", "µm", nullable=True),
    Column("nearest_mtoc_uid", "string"),
    Column("nearest_mtoc_dist_um", "float", "µm", nullable=True),
)

MTOCS = (
    DONOR, FOV, Column("mtoc_uid", "string"),
    Column("neuron_uid", "string"), Column("label", "int"), Column("neuron_id", "int"),
    Column("volume_um3", "float", "µm³"), Column("frac_in_soma", "float", "fraction"),
    Column("containment", "string"), Column("base_dist_um", "float", "µm", nullable=True),
    Column("centroid_z", "float", "voxels"), Column("centroid_y", "float", "voxels"),
    Column("centroid_x", "float", "voxels"),
    Column("dist_to_cilium_base_um", "float", "µm", nullable=True), Column("cilium_uid", "string"),
)

LINKS = (
    Column("src_type", "string"), Column("src_uid", "string"), Column("dst_type", "string"),
    Column("dst_uid", "string"), Column("relation", "string"), Column("distance_um", "float", "µm"),
    Column("threshold_um", "float", "µm", nullable=True), Column("method", "string"),
)

NEURON_TABLE = (
    DONOR, FOV, Column("neuron_uid", "string"),
    Column("neuron_id", "int"), Column("censoring", "string"), Column("eligible", "bool"),
    Column("has_cilium", "bool", nullable=True), Column("cilium_id", "int", nullable=True),
    Column("length_um", "float", "µm", nullable=True),
    Column("ac3_mean", "float", "intensity", nullable=True),
    Column("nearest_mtoc_dist_um", "float", "µm", nullable=True),
    Column("kept", "bool", nullable=True), Column("kept_reason", "string"), Column("status", "string"),
)

FOV_SUMMARY = (
    DONOR, FOV, Column("n_neurons", "int", "instances"),
    Column("n_eligible", "int", "instances"), Column("k_ciliated", "int", "instances"),
    Column("ciliated_wilson_lo", "float", "fraction", nullable=True),
    Column("ciliated_wilson_hi", "float", "fraction", nullable=True),
    Column("n_length", "int", "cilia"), Column("median_length_um", "float", "µm", nullable=True),
    Column("denominator", "string"), Column("status", "string"),
)

SCHEMAS = {
    "morphometry": MORPHOMETRY,
    "skeleton": SKELETON,
    "pcnt_morphometry": PCNT_MORPHOMETRY,
    "censoring": CENSORING,
    "cilia": CILIA,
    "neurons": NEURONS,
    "mtocs": MTOCS,
    "links": LINKS,
    "neuron_table": NEURON_TABLE,
    "fov_summary": FOV_SUMMARY,
}


@dataclass
class Fov:
    """One field of view: the image, its marker planes, its voxel size, the external masks, and
    where the two segmentations wrote."""

    image: Path
    series: int
    channels: dict[str, int]
    spacing_um: tuple[float, float, float]
    fov_id: str
    donor: str
    neun_instances: Path | None = None
    cilia_dir: Path | None = None
    mtoc_dir: Path | None = None


def load(image: Path, *, channels: Mapping[str, int], neun_instances: Path | None = None,
         series: int = 0, fov_id: str | None = None, donor: str | None = None) -> Fov:
    """Declare one fov, checking the files exist and reading the voxel size from the image."""
    image = Path(image).expanduser()
    if not image.is_file():
        raise FileNotFoundError(f"image: {image}")
    if neun_instances is not None:
        neun_instances = Path(neun_instances).expanduser()
        if not neun_instances.is_file():
            raise FileNotFoundError(f"neun_instances: {neun_instances}")
    stem = image.name
    for suffix in (".ome.tiff", ".ome.tif", ".lif", ".tif", ".tiff"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if str(image).lower().endswith((".ome.tif", ".ome.tiff")):
        import xml.etree.ElementTree as ET

        import tifffile
        with tifffile.TiffFile(str(image)) as tif:
            if not tif.is_ome or not tif.ome_metadata:
                raise ValueError(f"{image}: no ome-xml, so no voxel size")
            root = ET.fromstring(tif.ome_metadata)
        namespace = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        pixels: Any = root.find(f".//{namespace}Pixels")
        if pixels is None:
            raise ValueError(f"{image}: no ome Pixels element, so no voxel size")
        sizes = [pixels.get(f"PhysicalSize{axis}") for axis in "ZYX"]
        units = [pixels.get(f"PhysicalSize{axis}Unit") for axis in "ZYX"]
    elif image.suffix.lower() == ".lif":
        from readlif.reader import LifFile
        # readlif's scale is pixels per µm in (x, y, z) order
        scale: Any = LifFile(str(image)).get_image(series).scale
        sizes = [1.0 / float(scale[i]) if len(scale) > i and scale[i] else None for i in (2, 1, 0)]
        units = [None, None, None]
    else:
        raise ValueError(f"{image}: reads .lif and ome-tiff only")
    for axis, size, unit in zip("zyx", sizes, units, strict=True):
        if not size or float(size) <= 0.0:
            raise ValueError(f"{image}: no spacing on {axis}")
        # a silent factor of 1000 is what a declared unit exists to catch
        if unit not in (None, "µm", "um", "micron", "micrometer", "micrometre"):
            raise ValueError(f"{image}: spacing on {axis} is in {unit!r}, not µm")
    ident = fov_id or stem
    return Fov(image=image, series=int(series),
               channels={str(marker): int(index) for marker, index in channels.items()},
               spacing_um=(float(sizes[0]), float(sizes[1]), float(sizes[2])),  # type: ignore[arg-type]
               fov_id=ident, donor=donor or ident, neun_instances=neun_instances)


def read_plane(fov: Fov, marker: str) -> np.ndarray:
    """Read one channel of the fov by marker name, as stored."""
    index = fov.channels[marker]
    if str(fov.image).lower().endswith((".ome.tif", ".ome.tiff")):
        import tifffile
        with tifffile.TiffFile(str(fov.image)) as tif:
            axes = str(tif.series[0].axes)
            data = np.asarray(tif.series[0].asarray())
        if axes == "ZYX":
            data, axes = data[np.newaxis], "CZYX"
        if axes != "CZYX":
            raise ValueError(f"{fov.image}: expected ome axes CZYX or ZYX, got {axes!r}")
        if not 0 <= index < data.shape[0]:
            raise ValueError(f"{fov.image}: no plane {index}; the file has {data.shape[0]}")
        return np.asarray(data[index])
    from readlif.reader import LifFile
    lif: Any = LifFile(str(fov.image)).get_image(fov.series)
    if not 0 <= index < int(lif.channels):
        raise ValueError(f"{fov.image}: no plane {index}; series {fov.series} has "
                         f"{int(lif.channels)}")
    return np.stack([np.asarray(lif.get_frame(z=z, t=0, c=index))
                     for z in range(int(lif.dims.z))], axis=0)


def read_cmap(path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Read a .cmap's cell_body and nucleus instance maps as delivered, and its own excluded ids."""
    import h5py
    with h5py.File(path, "r") as handle:
        maps = [np.asarray(handle[f"{group}/data"][()]).astype(np.int32)  # type: ignore[index]
                for group in ("cell_body", "nucleus")]
        excluded = sorted({int(v) for v in handle["qc/excluded_ids"][()]})  # type: ignore[index]
    return maps[0], maps[1], excluded


def read_labels(path: Path) -> np.ndarray:
    """Read a label map written by write_labels."""
    import tifffile
    return np.asarray(tifffile.imread(path))


def write_labels(path: Path, labels: np.ndarray) -> Path:
    """Write a label map as an int32 tiff with deterministic bytes."""
    import tifffile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # photometric is pinned: tifffile would otherwise read a trailing axis of 3 or 4 as rgb
    tifffile.imwrite(str(path), np.asarray(labels).astype(np.int32), compression="zlib",
                     photometric="minisblack")
    return path


ACCEPTED_KINDS = {"int": "iu", "float": "fiu", "bool": "b", "string": "OUS"}

EMPTY_DTYPES = {("int", False): "int64", ("int", True): "Int64", ("float", False): "float64",
                ("float", True): "float64", ("bool", False): "bool", ("bool", True): "boolean",
                ("string", False): "str", ("string", True): "str"}


def frame_of(rows: Any, name: str) -> pd.DataFrame:
    """Build a table in its schema's column order, with the declared kinds when it has no rows."""
    columns = SCHEMAS[name]
    frame = pd.DataFrame(rows, columns=[c.name for c in columns])
    if frame.empty:
        frame = frame.astype({c.name: EMPTY_DTYPES[(c.kind, c.nullable)] for c in columns})
    return frame


def check_table(frame: pd.DataFrame, name: str, *, where: str) -> pd.DataFrame:
    """Return the frame if it has exactly the schema's columns, order, kinds and nulls."""
    columns = SCHEMAS[name]
    declared = [c.name for c in columns]
    problems: list[str] = []
    if list(frame.columns) != declared:
        missing = [c for c in declared if c not in frame.columns]
        extra = [c for c in frame.columns if c not in declared]
        problems.append(f"columns differ: missing {missing}, extra {extra}"
                        if missing or extra else f"columns out of order: {list(frame.columns)}")
    for column in columns:
        if column.name not in frame.columns:
            continue
        series: Any = frame[column.name]
        if getattr(series.dtype, "kind", "") not in ACCEPTED_KINDS[column.kind]:
            problems.append(f"{column.name}: declared {column.kind}, got {series.dtype}")
        if not column.nullable and len(series) and bool(series.isna().any()):
            problems.append(f"{column.name}: declared non-nullable, "
                            f"{int(series.isna().sum())} of {len(series)} rows are null")
    if problems:
        raise ValueError(f"{where}: not a {name} table\n  " + "\n  ".join(problems))
    return frame


def write_table(out_dir: Path, name: str, frame: pd.DataFrame) -> Path:
    """Check the frame against its schema and write it as <out_dir>/<name>.csv at full precision."""
    path = Path(out_dir) / f"{name}.csv"
    check_table(frame, name, where=f"write_table -> {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def read_table(out_dir: Path, name: str) -> pd.DataFrame:
    """Read <out_dir>/<name>.csv with round-trip floats and '' as the absent id, then check it."""
    path = Path(out_dir) / f"{name}.csv"
    # pandas' default parser reads 341 of B_02's 1500 volume_um3 values one ulp off the text
    frame = pd.read_csv(path, float_precision="round_trip")
    columns = SCHEMAS[name]
    strings = [c.name for c in columns
               if c.kind == "string" and not c.nullable and c.name in frame.columns]
    if strings:
        frame[strings] = frame[strings].fillna("").astype(str)
    # a csv cannot hold a null in an int or bool column, so a nullable one takes pandas' own kind
    nullable = {c.name: EMPTY_DTYPES[(c.kind, True)] for c in columns
                if c.nullable and c.kind in ("int", "bool") and c.name in frame.columns}
    if nullable:
        frame = frame.astype(nullable)
    if frame.empty:
        # a rowless csv reads back all-object, so no rows would also mean no schema
        frame = frame.astype({c.name: EMPTY_DTYPES[(c.kind, c.nullable)]
                              for c in columns if c.name in frame.columns})
    return check_table(frame, name, where=f"read_table <- {path}")


def write_run_json(out_dir: Path, *, fov: Fov, pipeline: str, values: Mapping[str, Any],
                   status: str, inputs: Mapping[str, str]) -> Path:
    """Write the record of one call: the fov, the values used, the weakest status, the inputs."""
    path = Path(out_dir) / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"pipeline": pipeline, "fov": asdict(fov), "values": dict(values),
              "status": status, "inputs": dict(inputs)}
    path.write_text(json.dumps(record, indent=2, default=str))
    return path
