# Code attachment for corl2019
This code example is for fully unsupervised training of objects 6d-pose.

## Step1 : Training pose network
The requirements of pose network are following:
1) ./data/[task_name]/[demo_name]/ depth and rgb image
2) ./output/segment/[task_name]/[demo_name] / segmentation mask
3) ./configure/[task_name].yaml

```
python3 ./main.py task1 pose 
```
or training continouously with existing weight
```
Python3 ./main.py task1 pose -c
```

To stop the training, you need to press ctrl+c
Then, weight will be automatically saved

## Step2 : Extracting the pose from the trained network

```
python3 ./main.py task1 pose -t
```
It saves pose trajectory in  './output/pose/[task_name]/se3_pose.npy' 

## Step3 : Visualized the trained result
```
python3 ./main.py task1 read_pose
```
It plots pose trajectory in './output/pose/read_pose/[task_name]'.
It plots pose projection on an image plane in './output/pose/read_pose2/[task_name]'.
