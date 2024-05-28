"""Module to generate sample data for testing purposes."""

from pathlib import Path

import dask_image.imread
import skimage.color
import skimage.data
import tifffile
from dask.array.core import Array
from loguru import logger


class SampleDaskStack:
    """
    Generate a sample 3D Dask stack for testing purposes.
    """

    def __init__(self, data_dir: Path, n_slices: int = 135) -> None:
        """
        Generate a sample Dask stack for testing purposes.

        Parameters
        ----------
        data_dir : Path
            Directory to save the sample stack.
            The 2D images will be saved in a subdirectory called "slices".
            The output zarr file will be saved in the root of the ``data_dir``
            as "stack.zarr".
        n_slices : int
            Number of slices to generate.

        """
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)

        self.n_slices = n_slices

        self.slice_dir = self.data_dir / "slices"
        self.slice_dir.mkdir(exist_ok=True)

        self.zarr_file_path = self.data_dir / "stack.zarr"

    def get_images(self) -> None:
        """
        Download the cat image write multiple 2D images to the slice directory.
        """
        # Check how many images are already written
        existing_images = list(self.slice_dir.glob("*.tif"))
        n_existing_images = len(existing_images)
        n_diff = n_existing_images - self.n_slices

        logger.info(f"Found {n_existing_images} existing images in {self.slice_dir}")

        if n_diff < 0:  # Need to generate more images
            logger.info(f"Generating {n_diff} missing images...")
            data_2d = skimage.color.rgb2gray(skimage.data.cat())
            for i in range(n_existing_images, self.n_slices):
                tifffile.imwrite(self.slice_dir / f"{str(i).zfill(3)}.tif", data_2d)
        elif n_diff > 0:  # Need to delete some images
            logger.info(f"Deleting {n_diff} extra images...")
            for i in range(self.n_slices, n_existing_images):
                (self.slice_dir / f"{str(i).zfill(3)}.tif").unlink()

    def generate_stack(self) -> Array:
        """
        Generate a 3D Dask stack from the 2D images in the slice directory.
        """
        stack = dask_image.imread.imread(str(self.slice_dir / "*.tif")).T
        logger.info(
            f"Read {stack.shape[2]} images into Dask array "
            f"with shape {stack.shape}, and chunk sizes {stack.chunksize}"
        )
        return stack
