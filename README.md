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
https://drive.google.com/open?id=1vLjtZ5F-90iEiidCS3EHYymBfYyUdstz <br />
1) download rgb-d dataset **'google_drive/data/[task_name]'** and unzip in **'./data/[task_name]'** <br />
2) downlaod semantic label **'google_drive/output/[task_name]'** and unzip in **'./output/[task_name]'** <br /> 

## Step1: Traning segmentation network
1) Execute python3 to train the segmentation network
```
python3 ./main.py [task_name] segment --train
```
- Check a log and figures created: **./log/[task_name]/segment_train.txt**, **./figure/[task_name]/segment/**

2) After the training converged, obtain the segmentation mask
```
python3 ./main.py [task_name] segment --test
```
- Check files created in **'./output/segment/[task_name]/'**

## Step2: Traning pose network
1) Execute python3 to train the pose network
```
python3 ./main.py [task_name] pose --train
```
- Check a log and figures created: **./log/[task_name]/pose_train.txt**, **./figure/[task_name]/pose/**

2) After the network converged, obtain the trained pose
```
python3 ./main.py [task_name] pose --test
```
- Check files created in **'./output/pose/[task_name]/'** 


## Step3 : Visualizing the trained result
To visualize the trained output, you need to exectue the visualizing code.
```
python3 ./main.py [task_name] read_pose
```
The pose trajectory will be plotted in **'./output/pose/read_pose/[task_name]'**. <br />
The pose projection on an imag will be plotted in **'./output/pose/read_pose2/[task_name]'**.


