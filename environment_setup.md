### commands for bioimage-fazallab build, run one line at a time in your terminal
```bash
conda create -y -n bioimage-fazallab -c conda-forge python=3
conda activate bioimage-fazallab  
#for windows users 
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
#for mac users 
conda install pytorch torchvision torchaudio -c pytorch-nightly


#to continue 
conda install conda-forge::ipykernel jupyter matplotlib 
conda install conda-forge::numpy pandas scikit-image scipy loguru
python -m pip install cellpose --upgrade
conda install conda-forge::napari matplotlib-scalebar
pip install bioio bioio-ome-tiff bioio-ome-zarr bioio-czi
conda install conda-forge::seaborn statannotations
```