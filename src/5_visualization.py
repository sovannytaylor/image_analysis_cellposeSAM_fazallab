"""Visualize feature information collected in prior script"""

import os
import numpy as np
import seaborn as sns
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
import functools
import cv2
from skimage import measure, segmentation, morphology
from scipy.stats import skewtest, skew
from skimage import morphology
from skimage.morphology import remove_small_objects
from statannotations.Annotator import Annotator
from loguru import logger
from matplotlib_scalebar.scalebar import ScaleBar
plt.rcParams.update({'font.size': 14})

input_folder = 'results/summary_calculations/'
plotting_folder = 'results/plotting/'


if not os.path.exists(plotting_folder):
    os.mkdir(plotting_folder)

### ---- Load in the excel sheet as a dataframe ---------

input_files = [f for f in os.listdir(input_folder) if f.endswith(('.xlsx', '.csv'))]
file_path = os.path.join(input_folder, input_files[0])

if file_path.endswith('.xlsx'):
    summary_df = pd.read_excel(file_path, engine='openpyxl')
elif file_path.endswith('.csv'):
    summary_df = pd.read_csv(file_path)
else:
    raise ValueError("Unsupported file format. Expected .xlsx or .csv.")
logger.info('file initialization complete :)')


### --- Visualize 

## I want to look at conditions for each peptide, comparing each condition against other conditions for that single peptide 

# Loop through each unique peptide
for peptide in summary_df['peptide'].unique():
    
    # Filter data for the current peptide
    summary_df_peptide = summary_df[summary_df['peptide'] == peptide]
    summary_df_reps_peptide = summary_df[summary_df['peptide'] == peptide]

    order = ['norm','norm_spike','LPD','LPD_spike','starv','starv_spike']
    x = 'condition'
   

    for parameter in ['mean_puncta_area', 'puncta_area_proportion', 'puncta_count',  'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'avg_eccentricity', 'cell_intensity_mean']:
        # plot data
        fig, ax = plt.subplots()
        ax = sns.stripplot(data=summary_df, x=x, y=parameter, order=order, dodge='True',
                        edgecolor='white', linewidth=1, size=8, alpha=0.4)
        sns.stripplot(data=summary_df, x=x, y=parameter, order=order, dodge='True',
                        edgecolor='k', linewidth=1, size=8, ax=ax)
        sns.boxplot(data=summary_df, x=x, y=parameter,
                    order=order, palette=['.7', '.8'], ax=ax)
        # statannot stats
        pairs = [(condition, 'norm') for condition in order if condition != 'norm']
        annotator = Annotator(ax, pairs, data=summary_df, x=x, y=parameter, order=order)
        annotator.configure(test='Mann-Whitney', verbose=2)
        annotator.apply_test()
        annotator.annotate()

        #add a title that also includes peptide iterated through 
        ax.set_title(f'{parameter} for {peptide}')
        # formatting
        sns.despine()
        plt.xticks(rotation=45)
        plt.xlabel('condition')
        plt.ylabel(parameter)
        plt.tight_layout()

        plt.savefig(f'{plotting_folder}2503B_{peptide}_{parameter}.png', format='png', dpi=300)
        plt.show()
