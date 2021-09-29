# Applications of AI and Computer Vision in Agriculture-Fruit recognition, localization and segmentation
- Utilise start-of-the-art CNN architectures technologies: Instance Segmentation to realise fruit recognition, localisation and segmentation in the farm, where the data is from open source dataset-ACFR farm Fruit Dataset collected at the farm in Warburton, Australia.

# Data Soruce
- https://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/ 
- Use the Apples data set

# Directory description

- mrcnn: main code files
- datasets: the data set intending to train
- samples/fruit: the codes for specified data sets, here is fruit data set
- logs: save the trained weights files

# How to run code

1. Enter the Mask_RCNN directory
2. Install dependencies

   ```bash
   pip3 install -r requirements.txt
   ```

2. Run setup.py
   ```bash
   python3 setup.py install
   ```
   
4. You can import the modules in Jupyter Notebook (see train_fruit.ipynb) or run it  from the command line:

   ```bash
    # First enter the Mask_RCNN/samples/fruit directory
    # Train a new model starting from pre-trained COCO weights
    python3 fruit.py train --dataset=./apples/ --weights=coco  --epoch=15
   
    # Resume training a model that you had trained earlier
    python3 fruit.py train --dataset=./apples/ --weights=last --epoch=25 --layers='all'
   
    # Train a new model starting from ImageNet weights
    python3 fruit.py train --dataset=./apples/ --weights=imagenet
    
    # Train a new model from a arbitrary pre-trained weights
    python3 fruit.py train --dataset=./apples/ --weights=path of .h5 files e.g. ./mask_rcnn_coco.h5 --epoch=11 --layers='all'
    
    # There are five arguments for command line: --dataset, --weights, --logs, --epoch, --layers, you can type: 
    python3 fruit.py --help
    # to see each parameter usage.
   ```

5. The inference code are ran on Google Gloud Colaboratory. First upload the Mask_RCNN folder to your google drive, then run the arbitrary .ipynb code file in Mask_RCNN/samples/fruit directory. 

# Reference
The code refers to https://github.com/matterport/Mask_RCNN.
