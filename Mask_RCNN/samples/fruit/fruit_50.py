"""
Mask R-CNN
Train on the ACFR Orchard Fruit dataset(https://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/)

Copyright (c) 2019 University of Southampton .
Licensed under the MIT License (see LICENSE for details)
Modified by Jing Meng
Reference to https://github.com/matterport/Mask_RCNN

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # First enter the Mask_RCNN/samples/fruit directory

    # Train a new model starting from pre-trained COCO weights
    python3 fruit_50.py train --dataset=./apples/ --weights=coco --epoch=15

    # Resume training a model that you had trained earlier
    python3 fruit_50.py train --dataset=./apples/ --weights=last --epoch=25 --layers='all'

    # Train a new model starting from ImageNet weights
    python3 fruit_50.py train --dataset=./apples/ --weights=imagenet --epoch=15
"""

import os
import sys
import json
import datetime
import time
import numpy as np
import csv
import skimage.draw

# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.path.abspath("./")  # currnet directory

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config_50 import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class FruitConfig(Config):
    """Configuration for training on the fruit dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "apple"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + fruit

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class FruitDataset(utils.Dataset):
    def load_fruit(self, dataset_dir, subset):
        """Load a subset of the fruit dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train,val or test
        """
        # Add classes. We have only one class to add.
        self.add_class("apple", 1, "apple")

        # Train, validation or test dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        print(dataset_dir)

        # Load annotations
        img_file_list = []
        ano_file_list = []
        for name in os.listdir(dataset_dir):
            if name.endswith(".png"):
                img_file_list.append(name)
            elif name.endswith(".csv"):
                ano_file_list.append(name)

        for name in img_file_list:
            index_ = name.index("_")
            pre_name = name[:index_ + 3].strip()
            #               print(pre_name)

            #               try:
            #                  ano_index=ano_file_list.index(pre_name)
            #               except:
            #                  continue

            if (pre_name + ".csv") not in ano_file_list:
                continue

            r_list = []
            c_list = []
            radius_list = []
            ano_path = os.path.join(dataset_dir, pre_name + ".csv")
            with open(ano_path) as f:  ####
                f_csv = csv.reader(f)
                headers = next(f_csv)
                for row in f_csv:
                    r_list.append(row[2])  ##### subscript, row is y, column is x
                    c_list.append(row[1])
                    radius_list.append(row[3])

            #               print("r_list:",r_list)
            #               print("c_list:",c_list)
            #               print("radius_list:",radius_list)

            image_path = os.path.join(dataset_dir, name)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            #               print("height:",height,"width:",width)   # 202*308

            self.add_image(
                "apple",
                image_id=name,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                r=r_list,
                c=c_list,
                radius=radius_list)  ######

    def load_mask(self, image_id):
        """Generate instance masks for an image.
           Returns:
           masks: A bool array of shape [height, width, instance count] with
              one mask per instance.
           class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a fruit dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "apple":
            return super(self.__class__, self).load_mask(image_id)

        # Convert circles to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        #           print("len(r):",len(info["r"]))
        #           print(info["r"])
        #           print(info["c"])
        #           print(info["radius"])
        mask = np.zeros([info["height"], info["width"], len(info["r"])],
                        dtype=np.uint8)
        #           print("mask_shape:",mask.shape)
        #           index=0
        for index in range(len(info["r"])):
            # Get indexes of pixels inside the circle and set them to 1
            #               print("r:",info["r"][index])
            #               print("c:",info["c"][index])
            #               print("radius:",info["radius"][index])
            rr, cc = skimage.draw.circle(float(info["r"][index]), float(info["c"][index]), float(info["radius"][index]),
                                         mask.shape)
            mask[rr, cc, index] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "apple":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model,epoch,layer):
    """Train the model."""
    # Training dataset.
    dataset_train = FruitDataset()
    dataset_train.load_fruit(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FruitDataset()
    dataset_val.load_fruit(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.

    t_start = time.time()
    image_ids = dataset_train.image_ids
    print("len(image_ids):",len(image_ids))
    if layer == 'heads':
        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=int(epoch),
                    layers='heads')  #########

    elif layer == '3+':
        # Training - Stage 3
        # Finetune layers from ResNet stage 2 and up
        print("Fine tune Resnet stage 3 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=int(epoch)+10,
                    layers='3+')

    elif layer == "4+":
        # Training - Stage 3
        # Finetune layers from ResNet stage 3 and up
        print("Fine tune Resnet stage 3 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=int(epoch) + 10,
                    layers='3+')

        # Training - Stage 4
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=int(epoch)+10*2,
                    layers='4+')

    elif layer == "all":
        # # Training - Stage 3
        # # Finetune layers from ResNet stage 2 and up
        # print("Fine tune Resnet stage 3 and up")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=int(epoch) + 10,
        #             layers='3+')
        #
        # # Training - Stage 4
        # # Finetune layers from ResNet stage 4 and up
        # print("Fine tune Resnet stage 4 and up")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=int(epoch) + 10 * 2,
        #             layers='4+')
        #
        # # Training - Stage 5
        # # Fine tune all layers
        # print("Fine tune all layers")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 10,
        #             epochs=int(epoch) + 10 * 3,
        #             layers='all')

        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=int(epoch),
                    layers='all')

    t_train = time.time() - t_start
    print("Total time: ", t_train)
    print("Training time: {} s. Average: {} s /image".format(
        t_train, t_train / len(image_ids)))

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect fruit.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="train")
    parser.add_argument('--dataset', required=False,
                        metavar="/dataset/apples/",
                        help='Directory of the Fruit dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--epoch', required=True,
                        help="Epoch of trainning")
    parser.add_argument('--layers', required=False, default= 'heads',choices=['heads','3+','4+','all'],
                        help="Layer for trainning")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Epoch: ", args.epoch)
    print("Layers: ", args.layers)

    # Configurations
    if args.command == "train":
        config = FruitConfig()
    else:
        class InferenceConfig(FruitConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)


    # Train or evaluate
    if args.command == "train":
        train(model,args.epoch,args.layers)
    else:
        print("'{}' is not recognized. "
              "Use 'train' ".format(args.command))
