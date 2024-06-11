# Yolov9 + Clip = Open-Vocabulary Object Detection

### Introduction
* This project is based on [Yolov9](https://github.com/WongKinYiu/yolov9), added with [CLIP](https://github.com/openai/CLIP), to realize Open-Vocabulary Object Detection without retraining the Yolov9 model.
* From the repositories above, you can download the necessary codes, datasets, etc.
* This repository **only** provides the `detect_ov.py` and `val_ov.py` used for OV detection.


### Quickstart
* run the instruction in the terminal
``` shell
python detect_ov.py --source './data/images/horses.jpg' --img 640 --device cpu --weights './yolov9-c-converted.pt' --name yolov9_ov
python val_ov.py --data data/objects365_val128.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device cpu --weights './yolov9-c-converted.pt' --save-json --workers 4 --name ovdetection
```
