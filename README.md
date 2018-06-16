# A3C
Asynchronous Actor Critic with unsupervised auxiliary tasks

## Done
	Updated small_maze maze to deepmind maze HEAD
	Maze is L-shaped at both ends. Visual results in navigation_visual_results folder
	Training results in tensorboard_results folder
	Auxiliary tasks
		rp - Reward prediction (skewed sampling)
		vp - Value prediction (unskewed sampling)
		rp_vp
		pc - Pixel control (unskewed sampling)
		rp_vp_pc
		fp - Frame prediction (unskewed sampling)
		rp_vp_fp
		ap - Action prediction (unskewed sampling)
		ftp - Frame threshold prediction (image is thresholded)
		flp - Flow prediction
        
    Bigger beta for exploration works best when the episode is short

## Setup

Requires deepmind lab. (Tested with last commit 832c50ee2a80b8b1e4a15fd60d1f8c1b7774c8ea)
```
$ git clone https://github.com/deepmind/lab.git
```

Place A3C folder in deepmind 'lab' folder
### Modify lab/BUILD
    py_binary(
    name = "a3c_train",
    srcs = ["A3C/main.py"],
    data = [":deepmind_lab.so"],
    main = "A3C/main.py",
    )
## Train model
### Modify parameters
in config.py
    
    ''' Choose task '''
    CONFIG = FP
    
### Run
from lab directory

    bazel run :a3c_train
	
## Visualize logs
from A3C directory
    
    tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'


## Example of output summaries
	tensorboard_results folder
	
## Navigation maps
	navigation_visual_results folder

## Done but not working as expected
1. Add action to LSTM 
2. FD - predict frame pixel difference instead of actual frame

[WORK IN PROGRESS]
