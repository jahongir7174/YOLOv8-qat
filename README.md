Quantization Aware Training Implementation of YOLOv8 without [DFL](https://ieeexplore.ieee.org/document/9792391) using PyTorch

### Installation

Execute the command:

```
conda create -n YOLO python=3.8
conda activate YOLO
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install opencv-python==4.5.5.64
pip install PyYAML
pip install tqdm
```

or 

```
pip3 install -r requirements.txt
```

### Train

* Configure your dataset path in `main.py` for training
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### Test

* Configure your dataset path in `main.py` for testing
* Run `python main.py --test` for testing

### Results

| Version | Epochs | Box mAP | CPU Latency |                   Download |
|:-------:|:------:|--------:|------------:|---------------------------:|
|  v8_n   |   20   |    33.4 |       13 ms | [model](./weights/best.ts) |
|  v8_n*  |  500   |    37.3 |       24 ms |                          - |
|  v8_s*  |  500   |    44.9 |           - |
|  v8_m*  |  500   |    50.2 |           - |
|  v8_l*  |  500   |    52.9 |           - |
|  v8_x*  |  500   |    53.9 |           - |

* `*` means that it is float precision, see reference

### Dataset structure

    ├── COCO 
        ├── images
            ├── train2017
                ├── 1111.jpg
                ├── 2222.jpg
            ├── val2017
                ├── 1111.jpg
                ├── 2222.jpg
        ├── labels
            ├── train2017
                ├── 1111.txt
                ├── 2222.txt
            ├── val2017
                ├── 1111.txt
                ├── 2222.txt

#### Reference

* https://github.com/ultralytics/yolov5
* https://github.com/ultralytics/ultralytics
