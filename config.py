import argparse

DEVICE = '/cpu:0'  # Create master network
NETWORK_TYPE = 'LSTM'
INPUT_SIZE = 84
NO_FRAMES = 1
GAMMA = 0.99
BETA = 0.01
LEARNING_RATE = 7e-4
EPISODES = 500
EXPERIENCE_BUFFER_MAXLEN = 2000
HAS_REWARD_PREDICTION = False
HAS_PIXEL_CONTROL = False
HAS_VALUE_PREDICTION = False
HAS_FRAME_PREDICTION = False
VP_LOSS_LAMBDA = 1
RP_LOSS_LAMBDA = 1
PC_LOSS_LAMBDA = 0.0001
FP_LOSS_LAMBDA = 1

# DEEPMIND LABYRINTH
GAME_NAME = 'LabMaze'

# PLE CATCHER
#GAME_NAME = 'Catcher'

# PLE RAYCASTMAZE
# GAME_NAME = 'Maze'

# PLE PIXELCOPTER
#GAME_NAME = 'Copter'

# Tasks
NO_AUX = 0
RP = 1
VP = 2
PC = 3
FP = 4
RP_VP = 5
RP_VP_PC = 6

CONFIG = FP

if GAME_NAME == 'Copter':
    ACTION_SIZE = 3
    BACKUP_STEP = 20
    EPISODES = 1000
    BETA = 1

    if CONFIG == RP:
        HAS_REWARD_PREDICTION = True
        RP_LOSS_LAMBDA = 0.01

    if CONFIG == VP:
        HAS_VALUE_PREDICTION = True
        VP_LOSS_LAMBDA = 0.001

    if CONFIG == PC:
        HAS_PIXEL_CONTROL = True
        PC_LOSS_LAMBDA = 0.00005

    if CONFIG == FP:
        HAS_FRAME_PREDICTION = True
        #-+FP_LOSS_LAMBDA = 0.0001
        #FP_LOSS_LAMBDA = 0.000001

    if CONFIG == RP_VP:
        HAS_REWARD_PREDICTION = True
        HAS_VALUE_PREDICTION = True
        RP_LOSS_LAMBDA = 0.01
        VP_LOSS_LAMBDA = 0.01

    if CONFIG == RP_VP_PC:
        HAS_REWARD_PREDICTION = True
        HAS_VALUE_PREDICTION = True
        HAS_PIXEL_CONTROL = True
        RP_LOSS_LAMBDA = 0.01
        VP_LOSS_LAMBDA = 0.01
        PC_LOSS_LAMBDA = 0.000005


if GAME_NAME == 'Maze':
    #TODO PARAMS NOT GOOD
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
    #TODO CATCHER CONFIG
    ACTION_SIZE = 3
    NO_FRAMES = 1
    BACKUP_STEP = 30
    EPISODES = 1000
    BETA = 0.01

    if CONFIG == RP:
        HAS_REWARD_PREDICTION = True
        RP_LOSS_LAMBDA = 0.01

    if CONFIG == VP:
        HAS_VALUE_PREDICTION = True
        VP_LOSS_LAMBDA = 0.001

    if CONFIG == PC:
        HAS_PIXEL_CONTROL = True
        PC_LOSS_LAMBDA = 0.00001

    if CONFIG == FP:
        HAS_FRAME_PREDICTION = True
        FP_LOSS_LAMBDA = 0.0001

    if CONFIG == RP_VP:
        HAS_REWARD_PREDICTION = True
        HAS_VALUE_PREDICTION = True
        VP_LOSS_LAMBDA = 0.01
        RP_LOSS_LAMBDA = 0.01

    if CONFIG == RP_VP_PC:
        HAS_REWARD_PREDICTION = True
        HAS_VALUE_PREDICTION = True
        HAS_PIXEL_CONTROL = True
        VP_LOSS_LAMBDA = 0.01
        RP_LOSS_LAMBDA = 0.01
        PC_LOSS_LAMBDA = 0.00001


#LAB
if GAME_NAME == 'LabMaze':
    IMPORT_LAB = True
    ACTION_SIZE = 6
    BACKUP_STEP = 20
    BETA = 1

    if CONFIG == NO_AUX:
        BETA = 1

    if CONFIG == RP:
        HAS_REWARD_PREDICTION = True
        RP_LOSS_LAMBDA = 1

    if CONFIG == VP:
        HAS_VALUE_PREDICTION = True
        VP_LOSS_LAMBDA = 0.1

    if CONFIG == PC:
        HAS_PIXEL_CONTROL = True
        PC_LOSS_LAMBDA = 0.0001 # not good enough

    if CONFIG == FP:
        HAS_FRAME_PREDICTION = True
        FP_LOSS_LAMBDA = 0.0001

    if CONFIG == RP_VP:
        HAS_REWARD_PREDICTION = True
        HAS_VALUE_PREDICTION = True
        VP_LOSS_LAMBDA = 0.1
        RP_LOSS_LAMBDA = 0.1


    if CONFIG == RP_VP_PC:
        HAS_REWARD_PREDICTION = True
        HAS_PIXEL_CONTROL = True
        HAS_VALUE_PREDICTION = True
        BETA = 0.01
        RP_LOSS_LAMBDA = 1
        VP_LOSS_LAMBDA = 1
        PC_LOSS_LAMBDA = 0.001




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
parser.add_argument('--episodes', help='Number of episodes',
                    default=EPISODES)
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