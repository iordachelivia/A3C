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
        
    Bigger beta for exploration works best when the episode is short

## Setup

Requires deepmind lab
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
from lab directory

    bazel run :a3c_train
	
## Visualize logs
from A3C directory
    
    tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'


## Example of output summaries
	tensorboard_results folder
	
## Navigation maps
	navigation_visual_results folder
	
	


## TODO

### Priority 1
1. Add action to LSTM
2. Stacked LSTM
3. Do not reset lstm state


### Priority 2
1. for FP save reconstructed images even though they are of no importance

### Priority 3
1. Saliency map
2. Attention mechanism
3. Foveal vision for navigation (https://arxiv.org/pdf/1801.08116.pdf)

[WORK IN PROGRESS]
