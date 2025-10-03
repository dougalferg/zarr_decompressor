# zarr\_decompressor

Hopefully a useful zarr decompressor for QCL IR datasets:

# Zarr Decompressor Toolkit



A simple Python toolkit to decompress uint16 Zarr arrays back to float32, designed for handling large hyperspectral datasets efficiently.



\## Installation



1\. Clone the repository:

`git clone https://github.com/dougalferg/zarr\_decompressor`

2\. Navigate into the directory:

`cd zarr\_decompressor`

3\. Install the required packages:

`pip install -r requirements.txt`



\## Usage



Here is how to use the toolkit to check and decompress a Zarr store that has been compressed.



```python

from zarr\_decompressor.toolkit import check\_source\_array, decompress\_zarr\_to\_memory



source\_file = r'path/to/your/compressed.zarr'



\# 1. Check if the source data is valid

if check\_source\_array(source\_file):

&nbsp;   # 2. Decompress the array into a NumPy array in memory

&nbsp;   decompressed\_data = decompress\_zarr\_to\_memory(source\_file)

&nbsp;   print("Data loaded successfully into memory.")

&nbsp;   print(decompressed\_data.shape)

