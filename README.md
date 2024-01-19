# Gaze Prototype

This is a very simple program to demonstrate the [Segment Anything](https://github.com/facebookresearch/segment-anything/) capability. 

## Installation
According to the Segment Anything documentation, the code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`.  This code has only been tested with 'python3.11', but should work on 'python3.8'.  NOTE: `python3.11` is the latest version to support Torch. 

To run this within a Python virtual environment (venv), create the venv:     
```$ python -m venv tobiiSAMenv```

Activate the venv     
```$ source tobiiSAMenv/bin/activate```

Install the dependencies:     
```$ pip install numpy```     
```$ pip install torch```     
```$ pip install matplotlib```     
```$ pip install opencv-python```     
```$ pip install segment-anything-py```     

## Getting Started
Download the model:
```$ mkdir model```    
```$ cd model```    
```$ curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth```    
```$ cd ...```    

Run the program     
```$ python predictor_example.py```

