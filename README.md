# Code attachment for corl2019
This code example is for fully unsupervised training of objects 6d-pose.

## Step1 : Training pose network
```
python3 ./main.py task1 pose 
```
or training continouously with existing weight
```
Python3 ./main.py task1 pose -c
```

To stop the training, you need to press ctrl+c
Then, weight will be automatically saved

## Step2 : Extracting pose from the trained network
```
Python3 ./main.py task1 pose -t
```
## Step3 : Visualized the trained result
```
Python3 ./main.py task1 read_pose
```

