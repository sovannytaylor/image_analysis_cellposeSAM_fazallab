"""Detect puncta, measure features, plot proofs (to ensure your analysis makes sense!)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools
import skimage as ski
from skimage import measure
from loguru import logger
from matplotlib_scalebar.scalebar import ScaleBar
plt.rcParams.update({'font.size': 14})

logger.info('Import ok')

input_folder = 'results/initial_cleanup/'
mask_folder = 'results/napari_masking/'
output_folder = 'results/summary_calculations/'
plotting_folder = 'results/plotting/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

if not os.path.exists(plotting_folder):
    os.mkdir(plotting_folder)

def feature_extractor(mask, properties=False):

    if not properties:
        properties = ['area', 'eccentricity', 'label', 'major_axis_length', 'minor_axis_length', 'perimeter', 'coords']

    return pd.DataFrame(ski.measure.regionprops_table(mask, properties=properties))


# ----------------Global variables----------------
# size of one pixel in µm
um_per_px = 0.0779907

# ----------------Initialise file list----------------
file_list = [filename for filename in os.listdir(
    input_folder) if 'npy' in filename]

images = {filename.replace('.npy', ''): np.load(
    f'{input_folder}{filename}') for filename in file_list}

masks = {masks.replace('_mask.npy', ''): np.load(
    f'{mask_folder}{masks}', allow_pickle=True) for masks in os.listdir(f'{mask_folder}') if '_mask.npy' in masks}

# make dictionary from images and masks array based on how many channels you have 
image_mask_dict = {
    key: np.stack([images[key][0, :, :], masks[key][0, :, :]])
    for key in masks
}

# # to add more channels from the image, follow example below
# image_mask_dict = {
#     key: np.stack([images[key][0, :, :], images[key][1, :, :], images[key][2, :, :], masks[key][0, :, :]])
#     for key in masks
# }

# ----------------collect feature information----------------
# remove saturated cells in case some were added during manual validation
logger.info('removing saturated cells')
not_saturated = {}
for name, image in image_mask_dict.items():
    labels_filtered = []
    # image order defined by image_mask_dict
    unique_val, counts = np.unique(image[-1, :, :], return_counts=True)

    # loop to remove saturated cells (>1% px values > 60000)
    for label in unique_val[1:]:
        pixel_count = np.count_nonzero(image[1, :, :] == label)
        # defining for which channel to check oversaturation, in this case it is channel 0
        cell = np.where(image[1, :, :] == label, image[0, :, :], 0)
        saturated_count = np.count_nonzero(cell == 65535)

        if (saturated_count/pixel_count) < 0.01:
            labels_filtered.append(label)

    cells_filtered = np.where(
        np.isin(image[1, :, :], labels_filtered), image[1, :, :], 0)

    # stack the filtered masks
    cells_filtered_stack = np.stack(
        (image[0, :, :], image[1, :, :], cells_filtered))
    not_saturated[name] = cells_filtered_stack

# now collect puncta and cell features info
logger.info('collecting feature info')
feature_information = []
for name, image in not_saturated.items():
    unique_val, counts = np.unique(image[1, :, :], return_counts=True)
    # find cell outlines for later plotting
    cell_binary_mask = np.where(image[1, :, :] !=0, 1, 0)
    contours = measure.find_contours(cell_binary_mask, 0.8)
    contour = [x for x in contours if len(x) >= 100]
    # loop to extract parameters from each cell
    for num in unique_val[1:]:
        # where the mask channel == cell number, give me the raw cell intensities and set all other pixels equal to 0
        cell = np.where(image[1, :, :] == num, image[0, :, :], 0)
        # collect intensity standard deviation and mean per cell
        cell_std = np.std(cell[cell != 0])
        cell_mean = np.mean(cell[cell != 0])
        # threshold puncta using standard deviation **may need to optimize number of stds**
        binary = (cell > (cell_std*4)).astype(int)
        # assign unique labels to puncta
        puncta_masks = measure.label(binary)
        # extract features per puncta
        cell_properties = feature_extractor(puncta_masks)
        # collate these features into one dataframe, then add important metadata as new columns
        properties = pd.concat([cell_properties])
        properties['image_name'] = name
        properties['cell_number'] = num
        properties['cell_size'] = np.size(cell[cell!=0])
        properties['cell_intensity_mean'] = cell_mean
        # filtering, keep puncta more than 9 pixels in area
        properties = properties[properties['area'] > 9]
        # filtering, keep cells less than 50,000 in mean pixel intensity
        properties = properties[properties['cell_intensity_mean'] < 50000]
        # add cell outlines for later proof plotting
        properties['cell_coords'] = [contour]*len(properties)
        # append properties to list
        feature_information.append(properties)
feature_information = pd.concat(feature_information)
logger.info('completed feature collection')

# ----------------average puncta properties per cell----------------
# --------Grab major and minor_axis_length for punctas--------
minor_axis = feature_information.groupby(
    ['image_name', 'cell_number']).mean(numeric_only=True)['minor_axis_length']
major_axis = feature_information.groupby(
    ['image_name', 'cell_number']).mean(numeric_only=True)['major_axis_length']
# --------Calculate average size of punctas per cell--------
avg_size = feature_information.groupby(
    ['image_name', 'cell_number']).mean(numeric_only=True)['area'].reset_index()
# --------Calculate average size of punctas per cell--------
avg_eccentricity = feature_information.groupby(
    ['image_name', 'cell_number']).mean(numeric_only=True)['eccentricity'].reset_index()
# --------Calculate proportion of area in punctas--------
cell_size = feature_information.groupby(
    ['image_name', 'cell_number']).mean(numeric_only=True)['cell_size']
puncta_area = feature_information.groupby(
    ['image_name', 'cell_number']).sum(numeric_only=True)['area']
puncta_proportion = ((puncta_area / cell_size) *
                   100).reset_index().rename(columns={'0': 'proportion_puncta_area'})
# --------Calculate number of 'punctas' per cell--------
puncta_count = feature_information.groupby(
    ['image_name', 'cell_number']).count()['area']
# --------Calculate puncta density (count/area in px) --------
puncta_density = (puncta_count / cell_size
                ).reset_index().rename(columns={'area': 'proportion_puncta_area'})
# --------Grab cell intensity mean --------
cell_intensity_mean = feature_information.groupby(
    ['image_name', 'cell_number']).mean(numeric_only=True)['cell_intensity_mean']

# ---------------- Summarize, save to csv ----------------
summary = functools.reduce(lambda left, right: pd.merge(left, right, on=['image_name', 'cell_number'], how='outer'), [avg_size, cell_size.reset_index(
), puncta_area.reset_index(), puncta_proportion, puncta_count.reset_index(), puncta_density, minor_axis, major_axis, cell_intensity_mean, avg_eccentricity])
summary.columns = ['image_name', 'cell_number', 'mean_puncta_area', 'cell_size', 'total_puncta_area',
                   'puncta_area_proportion', 'puncta_count', 'puncta_density', 'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'cell_intensity_mean', 'avg_eccentricity']

# save csv
summary.to_csv(f'{output_folder}puncta_detection_summary.csv')

# -------------- plotting proofs --------------
# plot proofs
for name, image in image_mask_dict.items():
    unique_val, counts = np.unique(image[-1, :, :], return_counts=True)

    # extract coords
    cell = np.where(image[1, :, :] != 0, image[0, :, :], 0)
    image_df = feature_information[(feature_information['image_name'] == name)]
    if len(image_df) > 0:
        cell_contour = image_df['cell_coords'].iloc[0]
        coord_list = np.array(image_df.coords)

        # plot
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image[0,:,:], cmap=plt.cm.gray_r)
        ax2.imshow(cell, cmap=plt.cm.gray_r)
        for cell_line in cell_contour:
            ax2.plot(cell_line[:, 1], cell_line[:, 0], linewidth=0.5, c='k')
        if len(coord_list) > 1:
            for puncta in coord_list:
                if isinstance(puncta, np.ndarray):
                    ax2.plot(puncta[:, 1], puncta[:, 0], linewidth=0.5)
        for ax in fig.get_axes():
            ax.label_outer()

        # Create scale bar, check the scaling factor from your metadata
        scalebar = ScaleBar(um_per_px, 'um', location = 'lower right', pad = 0.3, sep = 2, box_alpha = 0, color='k', length_fraction=0.3)
        ax1.add_artist(scalebar)

        # title and save
        fig.suptitle(name, y=0.78)
        fig.tight_layout()
        fig.savefig(f'{plotting_folder}{name}_proof.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)
        plt.close()