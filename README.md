# A3C
Asynchronous Actor Critic with unsupervised auxiliary tasks

# Done
	Updated small_maze script to deepmind maze HEAD
    Hyperparameters rp, vp, pc, fp, rp+vp, rp+vp+pc
        * Catcher
        * Pixelcopter
        * Lab maze ( pc is bad )
        
    Bigger beta for exploration works best when the episode is short
    
## TODO
### Priority 1

Redo frame prediction
Redo frame prediction with x[3] from seq
1. Trajectory(to see optimality of exploration)
    * catcher done
    * copter
    * maze (have to modify source code to get position)
    
2. AP : action prediction from experience replay
	2.1. no lstm
	2.2 conv 256->lstm 256->fc3 TODO : stride 1

3. Add action to LSTM
4. FR : frame reconstruction from latent space
5. Do not reset lstm state        
### Priority 2

1. frames are placed into experience buffer twice?
2. pc for lab maze : cannot find convergence hyperparams
3. for fp save reconstructed images even though they are of no importance


## Setup

Requires deepmind lab

Place A3C folder in deepmind 'lab' folder
Modify lab/BUILD
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


## Example of output summaries can be found in results folder    
[OLD. needs updating]

[WORK IN PROGRESS]
