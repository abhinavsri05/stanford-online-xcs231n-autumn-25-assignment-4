#!/bin/bash
wget http://cs231n.stanford.edu/imagenet_val_25.npz
if [ ! -d "coco_captioning" ]; then
    wget "http://cs231n.stanford.edu/coco_captioning.zip"
    unzip coco_captioning.zip
    rm coco_captioning.zip
fi