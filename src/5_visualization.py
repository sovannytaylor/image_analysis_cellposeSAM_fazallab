"""Visualize feature information collected in prior script"""

import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from loguru import logger
plt.rcParams.update({'font.size': 14})

input_folder = 'results/summary_calculations/'
plotting_folder = 'results/plotting/'

if not os.path.exists(plotting_folder):
    os.mkdir(plotting_folder)

# ---------------- load in excel sheet as dataframe ----------------
input_files = [f for f in os.listdir(input_folder) if f.endswith(('.xlsx', '.csv'))]
file_path = os.path.join(input_folder, input_files[0])

if file_path.endswith('.xlsx'):
    summary_df = pd.read_excel(file_path, engine='openpyxl')
elif file_path.endswith('.csv'):
    summary_df = pd.read_csv(file_path)
else:
    raise ValueError("Unsupported file format. Expected .xlsx or .csv.")
logger.info('file initialization complete :)')

# ---------------- wrangling ----------------
# add columns for sorting
# example of how to make a new column 
# this is saying take the df summary and add a column 'peptide'
# look in column 'image_name' and look at the first item as delimited by the '_' 
# example: 'date_PR30_lpd_63x_file'
# this would take PR30 because its the first item if you split by the '_'
summary_df['peptide'] = summary_df['image_name'].str.split('_').str[1]
# this would take the lpd
summary_df['condition'] = summary_df['image_name'].str.split('_').str[2]
# this is how you do it for a substring in between two delimiters 
summary_df['rep'] = summary_df['image_name'].str.split('_').str[-1].str.split('-').str[0]

# ---------------- visualize ----------------
# I want to compare conditions for each peptide
# Loop through each unique peptide
for peptide in summary_df['peptide'].unique():
    
    # Filter data for the current peptide
    summary_df_peptide = summary_df[summary_df['peptide'] == peptide]
    summary_df_reps_peptide = summary_df[summary_df['peptide'] == peptide]

    order = ['lpd']
    x = 'condition'

    for parameter in ['mean_puncta_area', 'puncta_area_proportion', 'puncta_count',  'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'avg_eccentricity', 'cell_intensity_mean']:
        # plot data
        fig, ax = plt.subplots()
        ax = sns.stripplot(data=summary_df, x=x, y=parameter, order=order, dodge='True', edgecolor='white', linewidth=1, size=8, alpha=0.4)
        sns.stripplot(data=summary_df, x=x, y=parameter, order=order, dodge='True',
                        edgecolor='k', linewidth=1, size=8, ax=ax)
        sns.boxplot(data=summary_df, x=x, y=parameter,
                    order=order, palette=['.7', '.8'], ax=ax)

        # # statannotation stats
        # pairs = [(condition, 'norm') for condition in order if condition != 'norm']
        # annotator = Annotator(ax, pairs, data=summary_df, x=x, y=parameter, order=order)
        # annotator.configure(test='Mann-Whitney', verbose=2)
        # annotator.apply_test()
        # annotator.annotate()

        #a dd a title that also includes peptide iterated through 
        ax.set_title(f'{parameter} for {peptide}')
        # formatting
        sns.despine()
        plt.xticks(rotation=45)
        plt.xlabel('condition')
        plt.ylabel(parameter)
        plt.tight_layout()

        plt.savefig(f'{plotting_folder}2503B_{peptide}_{parameter}.png', format='png', dpi=300)
        plt.show()
