# Code attachment for RAL/ICRA2020
This code is an example of fully unsupervised training of objects 6d-pose in a scene.

## Requirements:
python3 >= 3.6.8 <br />
<br />
(below are python3 packages) <br />
matplotlib >= 3.0.3 <br />
numpy >= 1.16.2 <br />
scipy >- 1.3.0 <br />
tensorflow-gpu == 1.13.1 <br />
tflearn >= 0.3.2 <br />

## Download dataset:
https://drive.google.com/drive/folders/1vLjtZ5F-90iEiidCS3EHYymBfYyUdstz?usp=sharing <br />

1) download rgb-d dataset ('./data/[task_name]' in google drive) and unzip in './data/[task_name]' <br />
(For example, './data/moving-a-block') <br />

2) downlaod semantic label ('./output/[task_name]' in google drive) and unzip in './output/[task_name]' <br /> 
(For example, './output/moving-a-block') <br />


## Step1 : Training pose network
The requirements for training the pose network are following:
1) ./data/[task_name]/[demo_name]/ depth and rgb image
2) ./output/segment/[task_name]/[demo_name] / segmentation mask
3) ./configure/[task_name].yaml

To train the pose network, please execute the code with python3:
```
python3 ./main.py task1 pose 
```
If you have pre-trained weight in './weight/pose/[task_name]/', you can use the command:
```
Python3 ./main.py task1 pose -c
```

The log file wile be saved in './log/[task_name]/pose_train.txt' <br />
The training figure will be saved in './figure/pose/[task_name]' <br />
To stop the training, you need to press ctrl+c. Then, weight will be automatically saved


## Step2 : Extracting the pose from the trained network
To extract the trained pose from the network, you need to execute the network in a test mode.
```
python3 ./main.py task1 pose -t
```
The pose trajectory will be saved in  './output/pose/[task_name]/se3_pose.npy' 

## Step3 : Visualizing the trained result
To visualize the trained output, you need to exectue the visualizing code.
```
python3 ./main.py task1 read_pose
```
The pose trajectory will be plotted in './output/pose/read_pose/[task_name]'. <br />
The pose projection on an imag will be plotted in './output/pose/read_pose2/[task_name]'.
