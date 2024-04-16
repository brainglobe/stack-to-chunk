"""
An example script to convert a multi-tiff stack to a chunked zarr file.

This script loads the multi-tiff stack from a Google Drive folder into a dask
array, and then saves it as a chunked zarr file.
"""

import os
import shutil
from pathlib import Path

import dask_image.imread
from loguru import logger

from stack_to_chunk import MultiScaleGroup

OVERWRITE_EXISTING_ZARR = True
USE_SAMPLE_DATA = False


def _load_env_var_as_path(env_var: str) -> Path:
    """Load an environment variable as a Path object."""
    path_str = os.getenv(env_var)

    if path_str is None:
        msg = f"Please set the environment variable {env_var}."
        raise ValueError(msg)

    path = Path(path_str)
    if not path.is_dir():
        msg = f"{path} is not a valid directory path."
        raise ValueError(msg)
    return path


# Paths to the Google Drive folder containing tiffs for all subjects & channels
# and the output folder for the zarr files (both set as environment variables)
input_dir = _load_env_var_as_path("ATLAS_PROJECT_TIFF_INPUT_DIR")
output_dir = _load_env_var_as_path("ATLAS_PROJECT_ZARR_OUTPUT_DIR")


if USE_SAMPLE_DATA:
    from stack_to_chunk.sample_data import SampleDaskStack

    cat_data = SampleDaskStack(output_dir / "sample_data", n_slices=1053)
    cat_data.get_images()
    chunk_size = 32

else:
    chunk_size = 128
    # Define subject ID and check that the corresponding folder exists
    subject_id = "topro35"
    assert (input_dir / subject_id).is_dir()

    # Define channel (by wavelength) and check that there is exactly one folder
    # containing the tiff files for this channel in the subject folder
    channel = "488"
    channel_dirs = sorted(input_dir.glob(f"{subject_id}/*{channel}*"))
    assert len(channel_dirs) == 1
    channel_dir = channel_dirs[0]


if __name__ == "__main__":
    if USE_SAMPLE_DATA:
        zarr_file_path = cat_data.zarr_file_path
        da_arr = cat_data.generate_stack()
    else:
        # Create a folders for the subject and channel in the output directory
        zarr_file_path = output_dir / subject_id / f"{subject_id}_{channel}.zarr"

        # Read the tiff stack into a dask array
        # Passing only the first tiff is enough
        # (because the rest of the stack is refererenced in metadata)
        tiff_files = sorted(channel_dir.glob("*.tif"))
        da_arr = dask_image.imread.imread(tiff_files[0]).T
        logger.info(
            f"Read tiff stack into Dask array with shape {da_arr.shape}, "
            f"dtype {da_arr.dtype}, and chunk sizes {da_arr.chunksize}"
        )

    # Delete existing zarr file if it exists and we want to overwrite it
    if OVERWRITE_EXISTING_ZARR and zarr_file_path.exists():
        logger.info(f"Deleting existing {zarr_file_path}")
        shutil.rmtree(zarr_file_path)

    if OVERWRITE_EXISTING_ZARR or not zarr_file_path.exists():
        # Create a MultiScaleGroup object (zarr group)
        zarr_group = MultiScaleGroup(
            zarr_file_path,
            name="stack",
            spatial_unit="micrometer",
            voxel_size=(0.98, 0.98, 3),
            interpolation="linear",
        )
        # Add full resolution data to zarr group 0
        zarr_group.add_full_res_data(
            da_arr,
            n_processes=5,
            chunk_size=chunk_size,
            compressor="default",
        )

        # Add downsampled levels
        # Each level corresponds to downsampling by a factor of 2**i
        for i in [1, 2]:
            zarr_group.add_downsample_level(i)
