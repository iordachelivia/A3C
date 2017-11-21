# A3C
Asynchronous Actor Critic

# TODO
* frames are placed into experience buffer twice?

    # Raycast
    * find shortest episode time for raycastmaze to converge
    # Pixelcopter
    * add support (should be appropriate for maze equivalent)





Requires deepmind lab

Place A3C folder in deepmind 'lab' folder

## Train model
	from lab directory run :
	bazel run //A3C:train --define headless=osmesa
	
## Visualize logs
	from A3C directory :
    tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'


## Example of output summaries can be found in results folder    
    
[WORK IN PROGRESS]
