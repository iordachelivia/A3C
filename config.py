import argparse

DEVICE = '/cpu:0'  # Create master network
NETWORK_TYPE = 'LSTM'
INPUT_SIZE = 84
NO_FRAMES = 1
GAMMA = 0.99
BETA = 0.01
LEARNING_RATE = 7e-4
EPISODES = 501
IS_TRAINING = True
EXPERIENCE_BUFFER_MAXLEN = 2000
# EXPERIENCE_BUFFER_MAXLEN = 40
''' Do not modify this portion '''
HAS_REWARD_PREDICTION = False
HAS_PIXEL_CONTROL = False
HAS_VALUE_PREDICTION = False

HAS_FRAME_PREDICTION = False
# predict diference between frames
HAS_FRAME_DIF_PREDICTION = False

HAS_FRAME_THRESH_PREDICTION = False

HAS_FLOW_PREDICTION = False

HAS_ACTION_PREDICTION = False

HAS_VQVAE_FRAME_RECONSTRUCTION = False



VP_LOSS_LAMBDA = 1
RP_LOSS_LAMBDA = 1
PC_LOSS_LAMBDA = 0.0001
FP_LOSS_LAMBDA = 1
FP_THRESH_LOSS_LAMBDA = 1
AP_LOSS_LAMBDA = 1
#FLOW
FL_LOSS_LAMBDA = 1
VQVAE_LOSS_LAMBDA = 1

# Tasks
NO_AUX = 0
RP = 1
VP = 2
PC = 3
FP = 4
FP_DIF = 5
FP_THRESH = 6
FL = 7
AP = 8
VQVAE = 9
RP_VP = 10
RP_VP_PC = 11
RP_VP_FP = 12
RP_VP_AP = 13


''' Choose env '''
# DEEPMIND LABYRINTH
GAME_NAME = 'LabMaze'

# PLE CATCHER
#GAME_NAME = 'Catcher'
# GAME_NAME = 'Catcher'

# PLE RAYCASTMAZE
# GAME_NAME = 'Maze'
#
# PLE PIXELCOPTER
#GAME_NAME = 'Copter'


''' Choose task '''
CONFIG = FP

# CONCAT ACTION IN LSTM
# only available for labmaze and FP task and RP+VP task and VP task
CONCAT_ACTION_LSTM = False

if GAME_NAME == 'Copter':
    CONCAT_ACTION_LSTM = False
    ACTION_SIZE = 3
    BACKUP_STEP = 20
    EPISODES = 1000
    BETA = 1
    if CONFIG == FL:
        HAS_FLOW_PREDICTION = True
        FL_LOSS_LAMBDA = 1
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
        FP_LOSS_LAMBDA = 0.0001
        
    if CONFIG == AP:
        HAS_ACTION_PREDICTION = True
        AP_LOSS_LAMBDA = 1
        #todo try(no tries)

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


if GAME_NAME == 'Catcher':
    ACTION_SIZE = 3
    NO_FRAMES = 1
    BACKUP_STEP = 30
    EPISODES = 1000
    BETA = 0.01
    #CONCAT_ACTION_LSTM = False

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

    # not optimized
    if CONFIG == FP_DIF:
        HAS_FRAME_DIF_PREDICTION = True
        HAS_FRAME_PREDICTION = True

        FP_LOSS_LAMBDA = 0.0001

    # not optimized
    if CONFIG == FP_THRESH:
        HAS_FRAME_THRESH_PREDICTION = True
        FP_THRESH_LOSS_LAMBDA = 1

    if CONFIG == FL:
        HAS_FLOW_PREDICTION = True
        FL_LOSS_LAMBDA = 1

    if CONFIG == AP:
        HAS_ACTION_PREDICTION = True
        AP_LOSS_LAMBDA = 1

    if CONFIG == VQVAE:
        HAS_VQVAE_FRAME_RECONSTRUCTION = True
        VQVAE_LOSS_LAMBDA = 1e-24

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
        CONCAT_ACTION_LSTM = False

    if CONFIG == RP:
        HAS_REWARD_PREDICTION = True
        RP_LOSS_LAMBDA = 1
        CONCAT_ACTION_LSTM = False

    if CONFIG == VP:
        HAS_VALUE_PREDICTION = True
        VP_LOSS_LAMBDA = 0.1

        #Not optimized
        if CONCAT_ACTION_LSTM:
            VP_LOSS_LAMBDA = 0.1
            VP_LOSS_LAMBDA = 0.05
            VP_LOSS_LAMBDA = 1.0
            VP_LOSS_LAMBDA = 0.2
            VP_LOSS_LAMBDA = 0.1


    if CONFIG == PC:
        HAS_PIXEL_CONTROL = True
        PC_LOSS_LAMBDA = 0.0001 # not good enough
        CONCAT_ACTION_LSTM = False

    if CONFIG == FP:
        HAS_FRAME_PREDICTION = True
        FP_LOSS_LAMBDA = 0.0001 #   BEST FOR K=1

        #different hyperparam
        #not optimized
        if CONCAT_ACTION_LSTM:
            FP_LOSS_LAMBDA = 0.0001
            FP_LOSS_LAMBDA = 0.00001
            FP_LOSS_LAMBDA = 0.001
            FP_LOSS_LAMBDA = 0.01
        
        #backward k=-1  
        FP_LOSS_LAMBDA = 0.0001
        
        #k=0
        #FP_LOSS_LAMBDA = 0.001
        #just conv
        #FP_LOSS_LAMBDA = 0.00001
        


    if CONFIG == FP_DIF:
        HAS_FRAME_DIF_PREDICTION = True
        HAS_FRAME_PREDICTION = True

        FP_LOSS_LAMBDA = 0.0001
        FP_LOSS_LAMBDA = 0.001
        FP_LOSS_LAMBDA = 0.00001
        FP_LOSS_LAMBDA = 0.000001
        FP_LOSS_LAMBDA = 0.01

    # not optimized
    if CONFIG == FP_THRESH:
        HAS_FRAME_THRESH_PREDICTION = True
        FP_THRESH_LOSS_LAMBDA = 0.0001
        FP_THRESH_LOSS_LAMBDA = 0.001
        FP_THRESH_LOSS_LAMBDA = 0.00007
        FP_THRESH_LOSS_LAMBDA = 0.0002
        FP_THRESH_LOSS_LAMBDA = 0.0001

        # FP_THRESH_LOSS_LAMBDA = 1 #softmax
        # FP_THRESH_LOSS_LAMBDA = 0.1
        # FP_THRESH_LOSS_LAMBDA = 0.01
        # FP_THRESH_LOSS_LAMBDA = 10
        
        FP_THRESH_LOSS_LAMBDA = 0.00001 #just conv

        FP_THRESH_LOSS_LAMBDA = 0.0001

    if CONFIG == FL:
        HAS_FLOW_PREDICTION = True
        FL_LOSS_LAMBDA = 1
        FL_LOSS_LAMBDA = 10
        FL_LOSS_LAMBDA = 100 # best for predicting next fm

        FL_LOSS_LAMBDA = 1e-3
        FL_LOSS_LAMBDA = 1e-5
        FL_LOSS_LAMBDA = 1e-6 # best for correct prediction
        FL_LOSS_LAMBDA = 1e-7
        FL_LOSS_LAMBDA = 1e-5

    if CONFIG == AP:
        HAS_ACTION_PREDICTION = True
        AP_LOSS_LAMBDA = 0.01
        CONCAT_ACTION_LSTM = False

    if CONFIG == VQVAE:
        HAS_VQVAE_FRAME_RECONSTRUCTION = True
        VQVAE_LOSS_LAMBDA = 1e-24
        VQVAE_LOSS_LAMBDA = 1e-22

        VQVAE_LOSS_LAMBDA = 1e-10
        VQVAE_LOSS_LAMBDA = 1e-15
        VQVAE_LOSS_LAMBDA = 1e-18

        # based on grad prob -17/-16??
        VQVAE_LOSS_LAMBDA = 1
        #VQVAE_LOSS_LAMBDA = 10


    if CONFIG == RP_VP:
        HAS_REWARD_PREDICTION = True
        HAS_VALUE_PREDICTION = True
        VP_LOSS_LAMBDA = 0.1
        RP_LOSS_LAMBDA = 0.1

        if CONCAT_ACTION_LSTM:
            VP_LOSS_LAMBDA = 0.1
            RP_LOSS_LAMBDA = 0.1


    if CONFIG == RP_VP_FP:
        HAS_REWARD_PREDICTION = True
        HAS_VALUE_PREDICTION = True
        HAS_FRAME_PREDICTION = True
        VP_LOSS_LAMBDA = 0.1
        RP_LOSS_LAMBDA = 0.1
        FP_LOSS_LAMBDA = 0.0001
        CONCAT_ACTION_LSTM = False


    if CONFIG == RP_VP_PC:
        HAS_REWARD_PREDICTION = True
        HAS_PIXEL_CONTROL = True
        HAS_VALUE_PREDICTION = True
        BETA = 0.01
        RP_LOSS_LAMBDA = 1
        VP_LOSS_LAMBDA = 1
        PC_LOSS_LAMBDA = 0.001
        CONCAT_ACTION_LSTM = False




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
parser.add_argument('--concat_action_lstm', help='Whether to concat action to '
                                                 'lstm or not',
                    default=CONCAT_ACTION_LSTM)
parser.add_argument('--has_reward_prediction', help='Use reward prediction as '
                    'an auxiliary task', default=HAS_REWARD_PREDICTION)
parser.add_argument('--has_pixel_control', help='Use pixel control as an '
                    'auxiliary task', default=HAS_PIXEL_CONTROL)
parser.add_argument('--has_value_prediction', help='Use value prediction as '
                    'an auxiliary task', default=HAS_VALUE_PREDICTION)
parser.add_argument('--has_frame_prediction', help='Use frame prediction as '
                    'an auxiliary task', default=HAS_FRAME_PREDICTION)
parser.add_argument('--has_frame_dif_prediction', help='Use frame diference '
                    'prediction as an auxiliary task',
                    default=HAS_FRAME_DIF_PREDICTION)
parser.add_argument('--has_frame_prediction_thresholded', help='Use frame '
                                                          'thresholded '
                    'prediction as an auxiliary task',
                    default=HAS_FRAME_THRESH_PREDICTION)
parser.add_argument('--has_flow_prediction', help='Use flow '
                    'prediction as an auxiliary task',
                    default=HAS_FLOW_PREDICTION)
parser.add_argument('--has_action_prediction', help='Use action prediction as '
                    'an auxiliary task', default=HAS_ACTION_PREDICTION)
parser.add_argument('--has_vqvae_prediction', help='Use vqvae prediction as '
                    'an auxiliary task', default=HAS_VQVAE_FRAME_RECONSTRUCTION)
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
parser.add_argument('--fp_thresh_loss_lambda', help='Frame '
                                                    'thresholded prediction '
                                                    'loss lambda '
                    'aux task', default=FP_THRESH_LOSS_LAMBDA)
parser.add_argument('--fl_loss_lambda', help='Flow prediction loss lambda '
                    'aux task', default=FL_LOSS_LAMBDA)
parser.add_argument('--ap_loss_lambda', help='Action prediction loss lambda '
                    'aux task', default=AP_LOSS_LAMBDA)
parser.add_argument('--vqvae_loss_lambda', help='VQVAE prediction loss lambda '
                    'aux task', default=VQVAE_LOSS_LAMBDA)
parser.add_argument('--is_training',help='training?',default=True)

FLAGS = parser.parse_args()
