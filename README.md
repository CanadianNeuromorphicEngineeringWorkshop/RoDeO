# RoDeO (Robot Detecting Objects) 

__Simulated object events.__ Objects are in conventional positions, perspective slightly from above (adjustable parameter), robot approaches objects from different sides. 35 ms samples. 5 instances of each class, 4 approaching directions, 3 perpectives for each object. Zoom-in zoom-out motion. Currently 868 data samples across the 4 classes.
- apple
- paper coffee cup
- mug
- water bottle (0,5 l)


### What's going on
- Script for generating event-based data ready. 

### Installation
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
