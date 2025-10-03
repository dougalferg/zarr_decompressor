from tqdm.auto import tqdm
import math
import itertools
import numpy as np
import zarr
from zarr.storage import LocalStore

OFFSET = 0.5535
FACTOR = np.float64(65535 / 45535 * 10000)

def transformToFloat(i_array):
    """Applies the inverse transformation to a NumPy array chunk."""
    f_array = i_array.astype(np.float32)
    f_array /= FACTOR
    f_array -= OFFSET
    return f_array
    
def check_source_dtype(source_array):
    """
    Checks if a given Zarr array has a uint16 dtype.
    Raises an exception if the check fails.
    (This is a refactored version of your original check_source_array)
    """
    if source_array.dtype != np.dtype('uint16'):
        raise ValueError(f"Array dtype is '{source_array.dtype}', but 'uint16' is required.")


def check_source_folders(source_path, group='0'):
    """
    Checks a Zarr group for required arrays under various possible names.

    It identifies the data, wavenumber, and optional mask arrays and returns
    their actual names.

    Args:
        source_path (str): Path to the root Zarr store.
        group (str): The name of the group to check.

    Returns:
        dict: A dictionary with the found names, e.g.,
              {'data': 'hyperspec', 'wavenumber': 'wvnm', 'mask': 'mask'}

    Raises:
        FileNotFoundError: If the essential data or wavenumber arrays cannot be found.
    """
    print(f"Searching for arrays in group '{group}'...")
    
    # Define the possible names for each array type
    data_aliases = ["hyperspec", "qcl_data", "data", "hyperspectral"]
    wavenumber_aliases = ["wvnm", "wavenumber", "wavenumbers", "wav"]
    
    try:
        z_root = zarr.open(source_path, mode='r')
        z_group = z_root[group]
        
        # Get a list of all array names in the group
        found_keys = list(z_group.keys())
        
        # Find the correct name for each required array
        data_name = next((name for name in data_aliases if name in found_keys), None)
        wavenumber_name = next((name for name in wavenumber_aliases if name in found_keys), None)
        mask_name = 'mask' if 'mask' in found_keys else None

        # Check if essential arrays were found
        if not data_name:
            raise FileNotFoundError(f"Could not find a data array. Searched for: {data_aliases}")
        if not wavenumber_name:
            raise FileNotFoundError(f"Could not find a wavenumber array. Searched for: {wavenumber_aliases}")
            
        # Check that the dtype is decompressable!
        print(f"   Verifying dtype for '{data_name}'...")
        check_source_dtype(z_group[data_name])
            
        print("Found required arrays:")
        print(f"   - Data: '{data_name}'")
        print(f"   - Wavenumber: '{wavenumber_name}'")
        if mask_name:
            print(f"   - Mask: '{mask_name}'")

        return {'data': data_name, 'wavenumber': wavenumber_name, 'mask': mask_name}

    except KeyError:
        raise FileNotFoundError(f"Group '{group}' not found in the Zarr store at '{source_path}'.")

def decompress_zarr_to_memory(source_path, group='0', array_name='hyperspec'):
    """
    Reads a specified array from a Zarr store chunk by chunk and returns a
    fully formed float32 NumPy array in memory.

    This version uses manual chunk iteration for maximum compatibility.

    Args:
        source_path (str): Path to the root Zarr store.
        group (str): The name of the group containing the array.
        array_name (str): The name of the array to decompress.

    Returns:
        np.ndarray: The final, fully decompressed float32 NumPy array.
    """
    print(f"Starting patch-wise decompression for '{group}/{array_name}'...")

    try:
        # Get the correct data names and run all checks in one go
        array_names = check_source_folders(source_path, group)
    except (FileNotFoundError, ValueError) as e:
        print("Error: Pre-decompression check failed. Halting execution.")
        raise e

    # Check it is actually a file that has been decompressed and 
    # Get the correct data names because i'm an idiot who wasn't consistent
    array_names = check_source_folders(source_path, group)

    # 1. Open the Zarr store and access the source array.
    z_root = zarr.open(source_path, mode='r')
    source_array = z_root[group][array_names['data']]
    shape = source_array.shape
    chunks = source_array.chunks
    
    print(f"Source: shape={shape}, chunks={chunks}, dtype={source_array.dtype}")

    # 2. Create the destination NumPy array in memory.
    dest_array = np.empty(shape, dtype='float32')
    print(f"Created destination NumPy array in memory with shape: {dest_array.shape}")

    # 3. Manually generate chunk slices and iterate.
    # This replaces the `source_array.blocks` iterator.
    print("\nProcessing chunks manually...")

    # Generate iterators for the start of each chunk along each dimension
    iter_d0 = range(0, shape[0], chunks[0])
    iter_d1 = range(0, shape[1], chunks[1])
    iter_d2 = range(0, shape[2], chunks[2])

    # Calculate total number of chunks for the progress bar
    total_chunks = math.ceil(shape[0]/chunks[0]) * math.ceil(shape[1]/chunks[1]) * math.ceil(shape[2]/chunks[2])

    # Create an iterator that yields the starting coordinate of each chunk
    chunk_starts = itertools.product(iter_d0, iter_d1, iter_d2)

    for d0_start, d1_start, d2_start in tqdm(chunk_starts, total=total_chunks, unit="chunk"):
        # Define the slice for the current chunk
        s = (
            slice(d0_start, min(d0_start + chunks[0], shape[0])),
            slice(d1_start, min(d1_start + chunks[1], shape[1])),
            slice(d2_start, min(d2_start + chunks[2], shape[2])),
        )
        
        # Read, transform, and write the chunk
        uint_chunk = source_array[s]
        float_chunk = transformToFloat(uint_chunk)
        dest_array[s] = float_chunk

    dest_wavenumber = z_root[group][array_names['wavenumber']][:]
    
    dest_mask = None
    if array_names['mask']:
        dest_mask = z_root[group][array_names['mask']][:]

    print("\nPatch-wise decompression to memory complete.")

    return dest_array, dest_wavenumber, dest_mask


def decompress_zarr_to_storage(source_path, dest_path, group='0', array_name='hyperspec'):
    """
    Reads a Zarr array chunk by chunk, decompresses it, and saves the
    result to a new Zarr store on disk.

    Args:
        source_path (str): Path to the source Zarr store (containing uint16).
        dest_path (str): Path to write the new destination Zarr store (for float32).
        group (str): The name of the group containing the array.
        array_name (str): The name of the array to decompress.
    """
    print(f"Starting disk-to-disk decompression for '{group}/{array_name}'...")
    
    try:
        # Get the correct data names and run all checks in one go
        array_names = check_source_folders(source_path, group)
    except (FileNotFoundError, ValueError) as e:
        print("Error: Pre-decompression check failed. Halting execution.")
        raise e

    
    # Check it is actually a file that has been decompressed and 
    # Get the correct data names because i'm an idiot who wasn't consistent
    array_names = check_source_folders(source_path, group)

    # Create zarr storage location
    store = LocalStore(dest_path)
    root = zarr.open(store, mode='w')
    gridIdx = '0'
    gridGroup = root.create_group(gridIdx)
    
    # 1. Open the source Zarr store and get array metadata.
    z_root = zarr.open(source_path, mode='r')
    source_array = z_root[group][array_names['data']]
    shape = source_array.shape
    chunks = source_array.chunks
    
    print(f"Source: shape={shape}, chunks={chunks}, dtype={source_array.dtype}")

    # 2. Create the destination Zarr store and array.
    # This pre-allocates the space on disk without loading data into memory.
    dest_array = gridGroup.create_dataset('qcl_data', shape=shape,
                                          chunks=chunks, dtype='float32')
    
    print(f"Destination created: {dest_path}")

    # 3. Manually iterate over chunks.
    print("\nProcessing and writing chunks...")
    iter_d0 = range(0, shape[0], chunks[0])
    iter_d1 = range(0, shape[1], chunks[1])
    iter_d2 = range(0, shape[2], chunks[2])
    total_chunks = math.ceil(shape[0]/chunks[0]) * math.ceil(shape[1]/chunks[1]) * math.ceil(shape[2]/chunks[2])
    chunk_starts = itertools.product(iter_d0, iter_d1, iter_d2)

    for d0_start, d1_start, d2_start in tqdm(chunk_starts, total=total_chunks, unit="chunk"):
        s = (
            slice(d0_start, min(d0_start + chunks[0], shape[0])),
            slice(d1_start, min(d1_start + chunks[1], shape[1])),
            slice(d2_start, min(d2_start + chunks[2], shape[2])),
        )
        
        uint_chunk = source_array[s]
        float_chunk = transformToFloat(uint_chunk)
        
        # Write the processed chunk to the destination Zarr array on disk.
        dest_array[s] = float_chunk

    wavenumber_data = z_root[group][array_names['wavenumber']][:]
    gridGroup.create_array('wavenumbers', data=wavenumber_data, chunks=wavenumber_data.shape)
    print("\nCopied wavenumber data.")

    if array_names['mask']:
        mask_data = z_root[group][array_names['mask']][:]
        # Use shape of mask for chunks, not wavenumber
        gridGroup.create_array('mask', data=mask_data, chunks=mask_data.shape)
        print("Copied mask data.")
    
    print(f"\nDecompression to '{dest_path}' complete.")