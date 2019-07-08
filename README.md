# Code attachment for Corl2019
This code is an example of fully unsupervised training of objects 6d-pose in a scene.

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

To stop the training, you need to press ctrl+c. Then, weight will be automatically saved


## Step2 : Extracting the pose from the trained network
To extract the trained pose from the network, you need to execute the network in a test mode.
```
python3 ./main.py task1 pose -t
```
It will save pose trajectory in  './output/pose/[task_name]/se3_pose.npy' 

## Step3 : Visualized the trained result
To visualize the trained output, you need to exectue the visualizing code.
```
python3 ./main.py task1 read_pose
```
It will plot pose trajectory in './output/pose/read_pose/[task_name]'. \\
It will plot pose projection on an image plane in './output/pose/read_pose2/[task_name]'.
