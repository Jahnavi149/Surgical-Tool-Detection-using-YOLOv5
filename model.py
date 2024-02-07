# Cloning YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5.git

!pip install -r yolov5/requirements.tx

# Dataset preparation
!mkdir dataset
!mkdir dataset/images
!mkdir dataset/labels


# Train-Test split
from sklearn.model_selection import train_test_split
import glob

image_files = glob.glob("dataset/images/*")
train_files, test_files = train_test_split(image_files, test_size=0.2)

# Save train and test files to disk
with open("dataset/train.txt", "w") as f:
    f.write("\n".join(train_files))

with open("dataset/test.txt", "w") as f:
    f.write("\n".join(test_files))

# Dataset configuration file
import yaml

data_yaml = dict(
    train = "dataset/train.txt",
    val = "dataset/test.txt",
    nc = 2,
    names = ['class1', 'class2']
)

with open("dataset.yaml", "w") as f:
    yaml.dump(data_yaml, f)

# Training YOLOv5 model
!python yolov5/train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --cfg yolov5s.yaml --weights yolov5s.pt --name my-experiment

# Testing YOLOv5 model
!python yolov5/detect.py --weights runs/train/my-experiment/weights/best.pt --img 640 --conf 0.4 --source dataset/test.txt

