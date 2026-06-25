# RoDeO (Robot Detecting Objects) 

## Data
__Selected objects:__
- apple
- paper coffee cup
- mug
- water bottle (0,5 l)
  
#### __Simulated object events__ 
Objects are in conventional positions, perspective slightly from above (adjustable parameter), robot approaches objects from different sides. 35 ms samples. 5 instances of each class, 4 approaching directions, 3 perpectives for each object. Zoom-in zoom-out motion. Currently 868 data samples across the 4 classes.  
- `data/events_converted_from 3D_models/{object}` includes `.npy` files with event streams (35 ms)  
- `data/events_converted_from 3D_models/events_converted_with_ROI/{object}` includes `.npz` files with event streams (35 ms), ROI center coordinates and ROI bounding box size. Code to read the .npz is included below.   

#### __Recorded DVS events__
Object recorded by DAVIS DVS 346 camera, 35 ms samples. The data are stored in `.npz` format and can be loaded using:
```
import numpy as np

NPZ_FILE = "data/recorded_DVS_data_with_ROI/bottle_0_00005_full.npz"   
W, H     = 346, 260   # input resolution (DAVIS346)

# ── load ──────────────────────────────────────────────────────────────────────
data     = np.load(NPZ_FILE, allow_pickle=True) 
ev       = data["events"].item() if data["events"].ndim == 0 else data["events"] #events inlcludeing x,y, timestamp and polarity
cx       = int(data["cx"].item()) #x coordinate of ROI center
cy       = int(data["cy"].item()) #y coordinate of ROI center
roi_size = int(data["roi_size"].item()) 
roi_half = roi_size // 2

print(f"Events: {len(ev)}  |  cx={cx} cy={cy}  |  roi_size={roi_size}")
print(f"Event fields: {ev.dtype.names}")

xs  = ev["x"].astype(np.int32) #x coordinates of events
ys  = ev["y"].astype(np.int32) #y coordinates of events
pol = ev["polarity"].astype(bool) 
t = ev["timestamp"].astype(np.int64)
```
There is also load + visualise script included in `data/read_from_npz.py`.  


## Installation
```bash
conda create -n snn_env python==3.12 pip
conda activate snn_env

# install latest pytorch with CUDA 12.8
pip install torch==2.9 torchvision==0.24 --index-url https://download.pytorch.org/whl/cu128

# install all important libraries
pip install -r requirements.txt

# install nengo dependencies
pip install nengo
pip install nengo-spa

# install sspspace library 
git clone https://github.com/ctn-waterloo/sspspace.git
cd sspspace/
python setup.py install

# setup jupyter notebook kernel - install nbclassic for better GUI
pip install nbclassic
python -m ipykernel install --user --name=snn_env
```
