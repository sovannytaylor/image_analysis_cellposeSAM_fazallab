import os
import numpy as np
from loguru import logger
from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter

logger.info('Import ok')

input_path = "P:/Sophie/Uptake/crotamine/04082025_2503C/2025-04-08"
output_folder = 'results/initial_cleanup/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# ---------------Initialize file_list---------------
# read in all folders with image files
valid_extensions = ['.czi', '.tif', '.tiff', '.lif']
file_list = [[f'{filename}' for filename in files if any(ext in filename.lower() for ext in valid_extensions)]
             for root, dirs, files in os.walk(f'{input_path}')]

# flatten file_list
file_list = [item for sublist in file_list for item in sublist]

# include anything that you don't want quantitated 
do_not_quantitate = []

# ---------------Collect image names & convert---------------
image_names = []
for filename in file_list:
    if all(word not in filename for word in do_not_quantitate):
        name = os.path.splitext(filename)[0]  # works for .czi, .tif, .lif
        image_names.append(name)

# remove duplicates
image_names = list(dict.fromkeys(image_names))

# ---------------Modified converter function---------------
def image_converter(image_name, input_folder, output_folder, tiff=False, array=True):
    """Convert and save images from supported formats"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    full_path = None
    for ext in valid_extensions:
        test_path = os.path.join(input_folder, image_name + ext)
        if os.path.exists(test_path):
            full_path = test_path
            break
    if full_path is None:
        logger.warning(f'File not found for {image_name}')
        return

    # Load image
    image = AICSImage(full_path).get_image_data("CYX", B=0, Z=0, V=0, T=0)

    if tiff:
        OmeTiffWriter.save(image, f'{output_folder}{image_name}.tif', dim_order='CYX')

    if array:
        np.save(f'{output_folder}{image_name}.npy', image)

# ---------------Run conversion---------------
for name in image_names:
    image_converter(name, input_folder=input_path, output_folder=output_folder)

logger.info('initial cleanup complete :)')