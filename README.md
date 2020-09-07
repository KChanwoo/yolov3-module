# YOLOv3 Module

YOLOv3 Module

## Installation
```
Conda 4.8.0 ~
Python 3.6.9 ~
Tensorflow 2.0

pip install tensorflow
pip install tensorflow-gpu
pip install git+https://git@github.com/KChanwoo/yolov3-module.git
```

## Use
```
from yolo import YOLO

# initialize
net = YOLO('path to save model')

# train
net.train('path to write logs')

# test
net.test('path to save prediction result', 'path to save ground truth information')

# predict
marked_image, detect_result = net.predict('image path')
```

## Publish
```
python setup.py bdist_wheel
```

## Author
ChanWoo Gwon, arknell@naver.com