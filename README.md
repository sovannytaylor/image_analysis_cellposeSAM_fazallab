# Image Analysis Pipeline 
This repository contains the analysis scripts that are associated with image analysis through segmentation via Cellpose SAM, as updated in May of 2025. The script has been adapted by Sovanny Taylor, originally receiving the scripts and logical workflow from Dr. Olivia Carmo and Dr. Dezerae Cox<sup id="a1">[1](#f1)</sup>.

## Analyses <b id="f20"></b>
### Fluorescence image analysis 
**Initial Cleanup Script** converts files from zeiss or leica (czi, tiff, lif files) instrumentation into numpy arrays for further analysis through python. 

**Cellpose and Napari Scripts** uses custom Python scripts, leveraging the Cellpose SAM<sup id="a2">[2](#f2)</sup> and napari<sup id="a3">[3](#f3)</sup> packages. Briefly, the logic follows as using specific channels to mask the whole cell, nuclei, or one or the other using Cellpose SAM. Segmentation should be manually inspected using napari to ensure the accuracy of masking assignment. This code also removes cells along the border and oversaturated cells based on channel set by the user. 

**Analysis_feature_information Script** collects feature information of cells and objects/ puncta found within cellular masks. These features were extracted using sci-kit image<sup id="a4">[4](#f4)</sup>. 


**Packages** Packages required for Python scripts can be accessed in the ```environment.yml``` file available in each analysis folder. To create a new conda environment containing all packages, run ```conda create -f environment.yml```


## Useful Resources 

Editing masks systematically before manual check using skikit-images<sup id="a5">[5](#f5)</sup>. This website outlines the Cell-poseSAM changes<sup id="a6">[6](#f6)</sup> that were apart of the 2025 update. This website is general information about the model<sup id="a7">[7](#f7)</sup>.

## References

<b id="f1">1.</b> This repository format adapted from https://github.com/dezeraecox-manuscripts/COX_Proteome-stability [↩](#a1)

<b id="f2">2.</b> Pachitariu M, Rariden M, Stringer C. Cellpose-SAM: superhuman generalization for cellular segmentation. biorxiv. 2025; doi:10.1101/2025.04.28.651001.[↩](#a2)

<b id="f3">3.</b> Sofroniew N, Lambert T, Evans K, Nunez-Iglesias J, Winston P, Bokota G, et al. napari/napari: 0.4.9rc2. Zenodo; 2021. doi:10.5281/zenodo.4915656. [↩](#a6)
updates: napari contributors (2019). napari: a multi-dimensional image viewer for python. doi:10.5281/zenodo.3555620 [↩](#a3)

<b id="f4">4.</b> Walt S van der, Schönberger JL, Nunez-Iglesias J, Boulogne F, Warner JD, Yager N, et al. scikit-image: image processing in Python. PeerJ. 2014;2: e453. doi:10.7717/peerj.453. [↩](#a4)

<b id="f5">5.</b> https://scikit-image.org/docs/0.25.x/api/skimage.morphology.html [↩](#a5)

<b id="f6">6.</b> https://cellpose.readthedocs.io/en/latest/settings.html [↩](#a6)

<b id="f7">7.</b> https://cellpose.readthedocs.io/en/latest/ [↩](#a7)
