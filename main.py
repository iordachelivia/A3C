import os 
import threading
import multiprocessing
from helper import *
from network import NetworkWrapper
from worker import Worker
import argparse
from time import sleep

INPUT_SIZE = 84
NO_FRAMES = 1
GAME_NAME = 'Catcher'
GAME_NAME='LabMaze'
GAME_NAME='Maze'


if GAME_NAME == 'Maze':
    ACTION_SIZE = 4
    NO_FRAMES = 1
    BACKUP_STEP = 30
    VP_LOSS_LAMBDA = 0.01
    RP_LOSS_LAMBDA = 1
    PC_LOSS_LAMBDA = 0.0001
    FP_LOSS_LAMBDA = 1
    HAS_REWARD_PREDICTION = True
    HAS_PIXEL_CONTROL = False
    HAS_VALUE_PREDICTION = True
    HAS_FRAME_PREDICTION = False

if GAME_NAME == 'Catcher':
    ACTION_SIZE = 3
    NO_FRAMES = 1
    BACKUP_STEP = 30
    VP_LOSS_LAMBDA = 0.01
    RP_LOSS_LAMBDA = 1
    PC_LOSS_LAMBDA = 0.0001
    FP_LOSS_LAMBDA = 1
    HAS_REWARD_PREDICTION = True
    HAS_PIXEL_CONTROL = False
    HAS_VALUE_PREDICTION = True
    HAS_FRAME_PREDICTION = False

#LAB
if GAME_NAME == 'LabMaze':
    IMPORT_LAB = True
    ACTION_SIZE = 6
    BACKUP_STEP = 20
    FP_LOSS_LAMBDA = 0.001

    #if all rp + vp + pc
    BACKUP_STEP = 20
    RP_LOSS_LAMBDA = 1
    VP_LOSS_LAMBDA = 1
    PC_LOSS_LAMBDA = 0.001

    #if just rp + vp
    #RP_LOSS_LAMBDA = 1
    #VP_LOSS_LAMBDA = 0.1


    HAS_REWARD_PREDICTION = False
    HAS_PIXEL_CONTROL = False
    HAS_VALUE_PREDICTION = False
    HAS_FRAME_PREDICTION = False


DEVICE = '/cpu:0'  # Create master network
NETWORK_TYPE = 'LSTM'

GAMMA = 0.99
BETA = 0.01
LEARNING_RATE = 7e-4

EXPERIENCE_BUFFER_MAXLEN = 2000

parser = argparse.ArgumentParser()
parser.add_argument('--input_size', help='Size of image (e.g 84 for 84x84)',
                    default=INPUT_SIZE)
parser.add_argument('--action_size', help ='Number of available actions',
                    default=ACTION_SIZE)
parser.add_argument('--no_frames', help='Number of frames (e.g 4 for Atari)',
                    default=NO_FRAMES)
parser.add_argument('--backup_step', help='Number of steps at which to backup '
                    'the gradients to the master', default=BACKUP_STEP)
parser.add_argument('--gamma', help='Gamma parameter', default=GAMMA)
parser.add_argument('--beta', help='Beta parameter', default=BETA)
parser.add_argument('--learning_rate', help='Learning rate',
                    default=LEARNING_RATE)
parser.add_argument('--game', help='Name of game (e.g Catcher',
                    default=GAME_NAME)
parser.add_argument('--device', help='Device to run on (only tested on cpu)',
                    default=DEVICE)
parser.add_argument('--network_type', help='Type of network (FF or LSTM)',
                    default=NETWORK_TYPE)
parser.add_argument('--has_reward_prediction', help='Use reward prediction as '
                    'an auxiliary task', default=HAS_REWARD_PREDICTION)
parser.add_argument('--has_pixel_control', help='Use pixel control as an '
                    'auxiliary task', default=HAS_PIXEL_CONTROL)
parser.add_argument('--has_value_prediction', help='Use value prediction as '
                    'an auxiliary task', default=HAS_VALUE_PREDICTION)
parser.add_argument('--has_frame_prediction', help='Use frame prediction as '
                    'an auxiliary task', default=HAS_FRAME_PREDICTION)
parser.add_argument('--experience_buffer_maxlen', help='Experience buffer '
                    'maximum length', default=EXPERIENCE_BUFFER_MAXLEN)
parser.add_argument('--pc_loss_lambda', help='Pixel control loss lambda '
                    'aux task', default=PC_LOSS_LAMBDA)
parser.add_argument('--vp_loss_lambda', help='Value prediction loss lambda '
                    'aux task', default=VP_LOSS_LAMBDA)
parser.add_argument('--rp_loss_lambda', help='Reward prediction loss lambda '
                    'aux task', default=RP_LOSS_LAMBDA)
parser.add_argument('--fp_loss_lambda', help='Frame prediction loss lambda '
                    'aux task', default=FP_LOSS_LAMBDA)

FLAGS = parser.parse_args()

load_model = False

#Reset the graph
tf.reset_default_graph()


dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = dir_path + '/model'
frames_path = dir_path + '/frames'
#Create folders
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(frames_path):
    os.makedirs(frames_path)

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
with tf.device(FLAGS.device),tf.Session(config = config) as sess:
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes',
                                  trainable=False)
    trainer = tf.train.AdamOptimizer(1e-4)
    #Create master network : it will hold the gradients
    #Each worker will update these gradients and sync with the master
    network_wrapepr = NetworkWrapper('global', trainer, None, FLAGS)
    master_network = network_wrapepr.get_network()

    # Set workers ot number of available CPU threads
    num_workers = multiprocessing.cpu_count()
    #num_workers = 1
    workers = []
    for index in range(num_workers):
        # Create worker classes
        worker = Worker(index, sess, trainer, dir_path, global_episodes,
                        master_network, FLAGS)
        # Initialize associated game
        worker.init_game(GAME_NAME, INPUT_SIZE)
        workers.append(worker)

    saver = tf.train.Saver(max_to_keep=5)

    coord = tf.train.Coordinator()
    if load_model == True:
        print ('LOG: Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # Start the work process for each worker
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)


'''
    COMMAND :bazel run //A3C_pc_latest:train --define headless=osmesa
    tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'
'''
