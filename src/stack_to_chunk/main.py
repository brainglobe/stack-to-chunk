"""
Main code for converting stacks to chunks.
"""

from math import ceil
from pathlib import Path
from typing import Any, Literal

import dask.array as da
import zarr
from dask import delayed
from dask.array.core import Array
from dask.diagnostics import ProgressBar
from loguru import logger
from numcodecs import blosc
from numcodecs.abc import Codec
from ome_zarr.dask_utils import resize

from stack_to_chunk._array_helpers import _copy_slab
from stack_to_chunk.ome_ngff import SPATIAL_UNIT


class MultiScaleGroup:
    """
    A class for creating and interacting with a OME-zarr multi-scale group.

    Parameters
    ----------
    path :
        Path to zarr group on disk.
    name :
        Name to save to zarr group.
    voxel_size :
        Size of a single voxel, in units of spatial_units.
    spatial_units :
        Units of the voxel size.
    interpolation :
            Interpolation method to use when downsampling data.

    """

    def __init__(
        self,
        path: Path,
        *,
        name: str,
        voxel_size: tuple[float, float, float],
        spatial_unit: SPATIAL_UNIT,
        interpolation: Literal["linear", "nearest"] = "linear",
    ) -> None:
        if path.exists():
            msg = f"{path} already exists"
            raise FileExistsError(msg)
        if interpolation not in ["linear", "nearest"]:
            msg = "interpolation must be 'linear' or 'nearest'"
            raise ValueError(msg)
        self._path = path
        self._name = name
        self._spatial_unit = spatial_unit
        self._voxel_size = voxel_size
        self._interpolation = interpolation

        self._create_zarr_group()

    def _create_zarr_group(self) -> None:
        """
        Create the zarr group.

        Saves a reference to the group on the ._group attribute.
        """
        self._group = zarr.open_group(store=self._path, mode="w")
        multiscales: dict[str, Any] = {}
        multiscales["version"] = "0.4"
        multiscales["name"] = self._name
        multiscales["axes"] = [
            {"name": "z", "type": "space", "unit": self._spatial_unit},
            {"name": "y", "type": "space", "unit": self._spatial_unit},
            {"name": "x", "type": "space", "unit": self._spatial_unit},
        ]
        multiscales["type"] = self._interpolation
        multiscales["metadata"] = {
            "description": f"Downscaled using {self._interpolation} resampling",
        }

        multiscales["datasets"] = []
        self._group.attrs["multiscales"] = multiscales
        self._add_multiscale_dataset_metadata(level=0)

    @property
    def levels(self) -> list[int]:
        """
        List of downsample levels currently stored.

        Level 0 corresponds to full resolution data, and level ``i`` to
        data downsampled by a factor of ``2**i``.
        """
        return [int(k) for k in self._group]

    def add_full_res_data(
        self,
        data: Array,
        *,
        chunk_size: int,
        compressor: Literal["default"] | Codec,
        n_processes: int,
    ) -> None:
        """
        Add the 'original' full resolution data to this group.

        Parameters
        ----------
        data :
            Input data. Must be 3D, and have a chunksize of ``(nx, ny, 1)``, where
            ``(nx, ny)`` is the shape of the input 2D slices.

        chunk_size :
            Size of chunks in output zarr dataset.
        compressor :
            Compressor to use when writing data to zarr dataset.
        n_processes :
            Number of parallel processes to use to read/write data.

        Raises
        ------
        RuntimeError :
            If full resolution data have already been added.

        """
        if "0" in self._group:
            msg = "Full resolution data already added to this zarr group."
            raise RuntimeError(msg)

        assert data.ndim == 3, "Input array is not 3-dimensional"
        if data.chunksize[2] != 1:
            msg = (
                f"Input array is must have a chunk size of 1 in the third dimension. "
                f"Got chunks: {data.chunksize}"
            )
            raise ValueError(msg)

        logger.info("Setting up copy to zarr...")
        slice_size_bytes = (
            data.nbytes // data.size * data.chunksize[0] * data.chunksize[1]
        )
        slab_size_bytes = slice_size_bytes * chunk_size
        logger.info(
            f"Each dask task will read ~{slab_size_bytes / 1e6:.02f} MB into memory"
        )

        self._group["0"] = zarr.create(
            data.shape,
            chunks=chunk_size,
            dtype=data.dtype,
            compressor=compressor,
        )

        nz = data.shape[2]
        slab_idxs: list[tuple[int, int]] = [
            (z, min(z + chunk_size, nz)) for z in range(0, nz, chunk_size)
        ]
        all_args = [
            (self._group["0"], data[:, :, zmin:zmax], zmin, zmax)
            for (zmin, zmax) in slab_idxs
        ]

        logger.info("Starting full resolution copy to zarr...")
        blosc_use_threads = blosc.use_threads
        blosc.use_threads = 0

        results = delayed([_copy_slab(*args) for args in all_args])
        with ProgressBar(dt=1):
            results.compute(num_workers=n_processes)

        blosc.use_threads = blosc_use_threads
        logger.info("Finished full resolution copy to zarr.")

    def add_downsample_level(self, level: int) -> None:
        """
        Add a level of downsampling.

        Parameters
        ----------
        level :
            Level of downsampling. Level ``i`` corresponds to a downsampling factor
            of ``2**i``.

        Notes
        -----
        To add level ``i`` to the zarr group, level ``i - 1`` must first have been
        added.

        """
        if not (level >= 1 and int(level) == level):
            msg = "level must be an integer >= 1"
            raise ValueError(msg)

        level_str = str(int(level))
        if level_str in self._group:
            msg = f"Level {level_str} already found in zarr group"
            raise RuntimeError(msg)

        level_minus_one = str(int(level) - 1)
        if level_minus_one not in self._group:
            msg = f"Level below (level={level_minus_one}) not present in group."
            raise RuntimeError(
                msg,
            )

        logger.info(f"Downsampling level {level_minus_one} to level {level_str}...")

        # Get the source data from the level below as a dask array
        source_group = self._group[level_minus_one]
        source_data = da.from_zarr(source_group, source_group.chunks)

        # Downsample the data by a factor of 2
        downsampled_shape = tuple(ceil(dim / 2) for dim in source_data.shape)
        spline_order = 1 if self._interpolation == "linear" else 0
        downsampled_data = resize(  # dask-ified wrapper around skimage.transform.resize
            source_data,
            output_shape=downsampled_shape,
            order=spline_order,
            preserve_range=True,  # should this be always True?
            anti_aliasing=True,  # should this be always True?
        )
        logger.info(
            f"Generated level {level_str} array with shape {downsampled_data.shape} "
            f"and chunk sizes {downsampled_data.chunksize}, using "
            f"{self._interpolation} interpolation."
        )

        # Write the downsampled data to the zarr group
        downsampled_store = self._group.require_dataset(
            level_str,
            shape=downsampled_data.shape,
            chunks=source_group.chunks,
            dtype=source_group.dtype,
            compressor=source_group.compressor,
        )
        downsampled_data.to_zarr(downsampled_store)

        self._add_multiscale_dataset_metadata(level)
        logger.info(f"Added level {level_str} to zarr group.")

    def _add_multiscale_dataset_metadata(self, level: int = 0) -> None:
        """
        Add the multiscale dataset metadata for the corresponding level to the group.

        Parameters
        ----------
        level :
            Level of downsampling. Level 0 corresponds to full resolution data.

        """
        scale_factors = [float(s * 2**level) for s in self._voxel_size]
        new_dataset = {
            "path": str(level),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": scale_factors,
                }
            ],
        }

        multiscales = self._group.attrs["multiscales"]
        existing_dataset_paths = [d["path"] for d in multiscales["datasets"]]
        if new_dataset["path"] not in existing_dataset_paths:
            multiscales["datasets"].append(new_dataset)
        self._group.attrs["multiscales"] = multiscales
