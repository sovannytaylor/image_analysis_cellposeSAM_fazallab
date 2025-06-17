# Image Analysis Pipeline 
This repository contains image analysis scripts adapted by Sovanny Taylor from Dr. Olivia M. S. Carmo and Dr. Dezerae Cox<sup id="a1">[1](#f1)</sup>.

## Analyses <b id="f20"></b>
### Fluorescence image analysis 
**Initial Cleanup** converts files from zeiss or leica instrumentation (e.g., czi, tiff, or lif files) into numpy arrays for further analysis. 

**Cellpose and napari** uses custom python scripts, relying primarily on cellposeSAM<sup id="a2">[2](#f2)</sup> and napari<sup id="a3">[3](#f3)</sup> packages. Briefly, fluorescence channels are used to segment the whole cell and/or nuclei using cellposeSAM. Segmentation should be manually inspected using napari to ensure the accuracy of the mask assignment. This code also removes cells along the border and oversaturated cells based on channel-of-interest as set by the user. 

**Analysis_feature_information** collects feature information of cells and objects (e.g., puncta) found within cellular masks. These features were extracted using sci-kit image<sup id="a4">[4](#f4)</sup>. 

**Packages** The required python packages can be accessed in the ```environment.yml```. Typically, the environment can be created on your device using ```conda env create --name <new_env_name> --file environment.yml```, however we have been running into bugs, so I recommend using the commands from ```environment_setup``` to make a suitable python environment.

## Advice for use
Please check off the tick boxes with an 'x' as you go.
- [ ] clone repo from github, giving it a FAIR name on your local machine (human and machine readable)
- [ ] uncomment raw_data and results folders in gitignore 
- [ ] delete placeholder files in raw_data and results folders
- [ ] upload raw data

## Useful Resources 
Editing masks systematically before manual check using skit-images<sup id="a5">[5](#f5)</sup>. This documentation<sup id="a6">[6](#f6)</sup> outlines the major cellposeSAM changes from the 2025 update, with more general information about the model here<sup id="a7">[7](#f7)</sup>.

## References
<b id="f1">1.</b> This repository format adapted from https://github.com/ocarmo/EMP1-trafficking_PTP7-analysis [↩](#a1)

<b id="f2">2.</b> Pachitariu M, Rariden M, Stringer C. Cellpose-SAM: superhuman generalization for cellular segmentation. biorxiv. 2025; doi:10.1101/2025.04.28.651001.[↩](#a2)

<b id="f3">3.</b> Sofroniew N, Lambert T, Evans K, Nunez-Iglesias J, Winston P, Bokota G, et al. napari/napari: 0.4.9rc2. Zenodo; 2021. doi:10.5281/zenodo.4915656. [↩](#a6)
updates: napari contributors (2019). napari: a multi-dimensional image viewer for python. doi:10.5281/zenodo.3555620 [↩](#a3)

<b id="f4">4.</b> Walt S van der, Schönberger JL, Nunez-Iglesias J, Boulogne F, Warner JD, Yager N, et al. scikit-image: image processing in Python. PeerJ. 2014;2: e453. doi:10.7717/peerj.453. [↩](#a4)

<b id="f5">5.</b> https://scikit-image.org/docs/0.25.x/api/skimage.morphology.html [↩](#a5)

<b id="f6">6.</b> https://cellpose.readthedocs.io/en/latest/settings.html [↩](#a6)

<b id="f7">7.</b> https://cellpose.readthedocs.io/en/latest/ [↩](#a7)
