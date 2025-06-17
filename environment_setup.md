### commands for bioimage-fazallab build, run one line at a time in your terminal
```bash
conda create -y -n napari-fazallab -c conda-forge python=3.9
conda activate napari-fazallab  
conda install -c conda-forge ipykernel jupyter matplotlib 
conda install -c conda-forge numpy pandas scikit-image scipy loguru
conda install conda-forge::aicsimageio
conda clean --all
# pytorch troubleshooting at https://discuss.pytorch.org/t/how-to-test-if-installed-torch-is-supported-with-cuda/11164/2
conda install pytorch torchvision -c pytorch 
conda install conda-forge::cellpose
conda install conda-forge::statannotations
conda install conda-forge::napari 
# aicsimageio downgraded 4.14.0-pyhd8ed1ab_0 --> 4.10.0-pyhd8ed1ab_0
pip install aicspylibczi>=3.0.5
pip install opencv-contrib-python # installed, but did not fix bug
conda install conda-forge::opencv # installed, but did not fix bug
pip install opencv-python
pip uninstall opencv-python 
conda remove opencv
pip install opencv-python 
conda install conda-forge::cellpose # install fine, but back to cv2 issue 'ImportError: DLL load failed while importing cv2: The specified module could not be found.'
```

### 2025-06-14
```bash
conda create -n bioimage-fazallab pytorch=1.8.2 cudatoolkit=10.2 -c pytorch-lts
conda activate bioimage-fazallab  
pip install cellpose
conda install conda-forge::ipykernel jupyter matplotlib 
conda install conda-forge::loguru
conda install conda-forge::scikit-image
conda install conda-forge::seaborn pandas statannotations
# TODO
conda install conda-forge::aicsimageio
pip install aicspylibczi>=3.0.5
conda install conda-forge::napari 
# aicsimageio downgraded 4.14.0-pyhd8ed1ab_0 --> 4.10.0-pyhd8ed1ab_0

```

### 2025-06-16
```bash
conda create -y -n bioimage-fazallab2 -c conda-forge python=3
conda activate bioimage-fazallab2  
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge ipykernel jupyter matplotlib 
conda install -c conda-forge numpy pandas scikit-image scipy loguru
python -m pip install cellpose --upgrade
conda install -c conda-forge napari # left out aicsimage for now
conda install conda-forge::matplotlib-scalebar
conda install conda-forge::seaborn statannotations
conda install conda-forge::aicsimageio # nope, circular import issue
pip install aicspylibczi>=3.0.5 # did not help
conda install napari-aicsimageio -c conda-forge # did not help
pip install dask[array] # all requirements already satisfied
conda install conda-forge::dask
pip install --upgrade dask numpy
# received this error
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# aicsimageio 4.10.0 requires lxml<5,>=4.6, but you have lxml 5.4.0 which is incompatible.
# aicsimageio 4.10.0 requires numpy<2,>=1.16, but you have numpy 2.3.0 which is incompatible.
# numba 0.61.2 requires numpy<2.3,>=1.24, but you have numpy 2.3.0 which is incompatible.
pip install bioio bioio-ome-tiff bioio-ome-zarr
# TODO
```

### 2025-06-16
```bash
conda create -y -n bioimage-fazallab3 -c conda-forge python=3
conda activate bioimage-fazallab3  
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::ipykernel jupyter matplotlib 
conda install conda-forge::numpy pandas scikit-image scipy loguru
python -m pip install cellpose --upgrade
conda install conda-forge::napari matplotlib-scalebar
pip install bioio bioio-ome-tiff bioio-ome-zarr bioio-czi
conda install conda-forge::seaborn statannotations
# TODO
```
