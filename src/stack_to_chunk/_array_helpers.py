import numpy as np
import numpy.typing as npt
import zarr
from dask import delayed
from scipy.ndimage import zoom


@delayed  # type: ignore[misc]
def _copy_slab(
    arr_zarr: zarr.Array, slab: npt.NDArray[np.uint16], zstart: int, zend: int
) -> None:
    """
    Copy a single slab of data to a zarr array.

    Parameters
    ----------
    arr_zarr :
        Array to copy to.
    slab :
        Slab of data to copy.
    zstart, zend :
        Start and end indices to copy to in destination array.

    """
    # Write out data
    arr_zarr[:, :, zstart:zend] = slab


def _downsample_and_copy_block(
    arr_zarr: zarr.Array,
    block: npt.NDArray[np.uint16],
    target_idxs: tuple[int, int, int],
) -> None:
    """
    Downsample and copy a single block of data to a zarr array.

    The block is downsampled by a factor of 2 using linear interpolation.
    The downsampled block is then copied to the zarr array, starting at the
    target indices.

    Parameters
    ----------
    arr_zarr :
        Array to copy to.
    block :
        Block of data to downsample and copy.
    target_indices :
        Start indices of the block in the destination zarr array.

    """
    # Downsample block by 2 using linear interpolation
    new_block = zoom(block, zoom=0.5, order=1, mode="nearest")
    assert all(
        new_block.shape[i] == block.shape[i] // 2 for i in range(3)
    ), "Downsampled block has incorrect shape"
    # Write out data
    extents = [
        slice(idx, idx + new_block.shape[i]) for i, idx in enumerate(target_idxs)
    ]
    arr_zarr[extents[0], extents[1], extents[2]] = new_block
