# zarr\_decompressor

Hopefully a useful zarr decompressor for QCL IR datasets:

# Zarr Decompressor Toolkit



A simple Python toolkit to decompress uint16 Zarr arrays back to float32, designed for handling large hyperspectral datasets efficiently.



\## Installation



1\. Clone the repository:

`git clone https://github.com/dougalferg/zarr_decompressor`

2\. Navigate into the directory:

`cd zarr_decompressor`

3\. Install the required packages:

`pip install -r requirements.txt`



\## Usage



Here is how to use the toolkit to check and decompress a Zarr store that has been compressed.



```python

from zarr_decompressor.toolkit import check_source_folders, decompress_zarr_to_memory



source_file = r'path/to/your/compressed.zarr'



# 1. Check if the source data is valid

if check_source_folders(source_file):

&nbsp;   # 2. Decompress the array into a NumPy array in memory

&nbsp;   decompressed_data = decompress_zarr_to_memory(source_file)

&nbsp;   print("Data loaded successfully into memory.")

&nbsp;   print(decompressed_data.shape)

