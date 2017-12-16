# A3C
Asynchronous Actor Critic with unsupervised auxiliary tasks

# Done
    Hyperparameters rp, vp, pc, fp, rp+vp, rp+vp+pc
        * Catcher
        * Pixelcopter
        * Lab maze ( pc is bad )
        
    Bigger beta for exploration works best when the episode is short
    
## TODO
### Priority 1

1. Trajectory(to see optimality of exploration)
    * catcher done
    * copter
    * maze (have to modify source code to get position)
    
2. AP : action prediction from experience replay
3. Add action to LSTM
4. FR : frame reconstruction from latent space
        
### Priority 2

1. frames are placed into experience buffer twice?
2. pc for lab maze : cannot find convergence hyperparams
3. for fp save reconstructed images even though they are of no importance


## Setup

Requires deepmind lab

Place A3C folder in deepmind 'lab' folder

## Train model
from lab directory

    bazel run //A3C:train --define headless=osmesa
	
## Visualize logs
from A3C directory
    
    tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'


## Example of output summaries can be found in results folder    
[OLD. needs updating]

[WORK IN PROGRESS]
