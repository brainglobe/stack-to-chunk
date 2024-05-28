"""
An example script to convert a multi-tiff stack to a chunked zarr file.

This script loads the multi-tiff stack from a Google Drive folder into a dask
array, and then saves it as a chunked zarr file.
"""

import os
import shutil
from pathlib import Path

from brainglobe_utils.IO.image import read_with_dask
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
    # Define subject ID which is part of the folder name
    subject_id = "ToPro54"
    # Define channel (by wavelength) and check that there is exactly one folder
    # with this subject ID and channel combo.
    channel = "561"
    channel_dirs = sorted(input_dir.glob(f"*{subject_id}_{channel}*"))
    assert len(channel_dirs) == 1, (
        f"Found {len(channel_dirs)} folders with subject ID {subject_id} and "
        f"channel {channel}. Please ensure there is exactly one such folder."
    )
    channel_dir = channel_dirs[0]
    logger.debug(f"Will read tiffs from {channel_dir}")


if __name__ == "__main__":
    if USE_SAMPLE_DATA:
        zarr_file_path = cat_data.zarr_file_path
        da_stack = cat_data.generate_stack()
    else:
        # Create a folders for the subject and channel in the output directory
        zarr_file_path = output_dir / subject_id / f"{subject_id}_{channel}.zarr"

        # Read the tiff stack into a dask array
        # (passing first image is enough, because it contains metadata about the stack)
        da_stack = read_with_dask(channel_dir).T
        logger.debug(
            f"Read tiff stack into Dask array with shape {da_stack.shape}, "
            f"dtype {da_stack.dtype}, and chunk sizes {da_stack.chunksize}"
        )

    # Delete existing zarr file if it exists and we want to overwrite it
    if OVERWRITE_EXISTING_ZARR and zarr_file_path.exists():
        logger.debug(f"Deleting existing {zarr_file_path}")
        shutil.rmtree(zarr_file_path)
        logger.debug(f"Deleted {zarr_file_path}")

    if OVERWRITE_EXISTING_ZARR or not zarr_file_path.exists():
        # Create a MultiScaleGroup object (zarr group)
        zarr_group = MultiScaleGroup(
            zarr_file_path,
            name="stack",
            spatial_unit="micrometer",
            voxel_size=(0.98, 0.98, 3),
        )
        # Add full resolution data to zarr group 0
        logger.debug("Adding full resolution data to zarr group 0...")
        zarr_group.add_full_res_data(
            da_stack,
            n_processes=8,
            chunk_size=chunk_size,
            compressor="default",
        )
        logger.debug("Finished adding full resolution data.")

        # Add downsampled levels
        # Each level corresponds to downsampling by a factor of 2**i
        for i in [1, 2]:
            logger.debug(f"Adding downsample level {i}...")
            zarr_group.add_downsample_level(i, n_processes=8)
            logger.debug(f"Finished adding downsample level {i}.")
