from vizdoom import *
import scipy
import numpy as np
from ple.games.catcher import Catcher
from ple import PLE
import random
#import gym
import deepmind_lab
import cv2
seed = 147
random.seed(seed)
import os
import sys
unbuffered = os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stdout = unbuffered
class LabWrapper:
    def __init__(self, width):
        self.game = None
        self.actions = None
        self.width = width
        self.frames_played = 0
        self.max_game_len = 3000

        # Create game env
        self.game = self.set_lab_game_setup()
        # Reset game
        self.restart_game()
    def _action(self,*entries):
        return np.array(entries, dtype=np.intc)

    def set_lab_game_setup(self):
        level = 'nav_maze_static_01'
        level = 'small_maze'
        #level = 'small_maze_multimap'
        env = deepmind_lab.Lab(
            level,
            ['RGB_INTERLACED'],
            config={
                'fps': str(60),
                'width': str(self.width),
                'height': str(self.width)
            })    

        self.actions = [
            self._action(-20, 0, 0, 0, 0, 0, 0),  # look_left
            self._action(20, 0, 0, 0, 0, 0, 0),  # look_right
            # _action(  0,  10,  0,  0, 0, 0, 0), # look_up
            # _action(  0, -10,  0,  0, 0, 0, 0), # look_down
            self._action(0, 0, -1, 0, 0, 0, 0),  # strafe_left
            self._action(0, 0, 1, 0, 0, 0, 0),  # strafe_right
            self._action(0, 0, 0, 1, 0, 0, 0),  # forward
            self._action(0, 0, 0, -1, 0, 0, 0),  # backward
            # _action(  0,   0,  0,  0, 1, 0, 0), # fire
            # _action(  0,   0,  0,  0, 0, 1, 0), # jump
            # _action(  0,   0,  0,  0, 0, 0, 1)  # crouch
        ]
        return env

    def restart_game(self):
        self.game.reset()
        self.frames_played = 0
        #the starting point is random through env

    def process_frame(self, image):
        image = image.astype(np.float32)
        #normalize
        image = np.reshape(image, [np.prod(image.shape), 1]) / 255.0
        return image

    def get_frame(self):
        colour_frame = self.game.observations()['RGB_INTERLACED']
        frame = cv2.cvtColor( colour_frame, cv2.COLOR_RGB2GRAY )
        #cv2.imshow('Image',self.last_frame)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return frame, colour_frame

    def game_finished(self):
        if self.frames_played >= self.max_game_len:
            print('LOG: max frames')
            return True
        if not self.game.is_running():
            print('LOG: game ended from engine')
        return not self.game.is_running()

    def make_action(self, action_index):
        #print ('ACTION index %d  action %s'%(action_index,str(self.actions[action_index])))
        reward = self.game.step(self.actions[action_index], num_steps=4)
        self.frames_played += 1

        return reward

''' Doom game class'''
class DoomWrapper:
    def __init__(self, width):
        '''
            @width : width of game window
        '''
        self.game = None
        self.max_game_len = 300
        self.actions = [[True, False, False], [False, True, False], [False, False, True]]
        self.width = width

        #Create game env
        self.game = self.set_doom_game_setup(self.max_game_len)

    def set_doom_game_setup(self, max_game_len):
        '''
            @max_game_len : maximum time steps allowed before terminating episode
        '''
        game = DoomGame()
        game.set_doom_scenario_path("basic.wad")  # This corresponds to the simple task we will pose our agent
        game.set_doom_map("map01")
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.add_available_button(Button.MOVE_LEFT)
        game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.ATTACK)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.set_episode_timeout(max_game_len)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        game.set_mode(Mode.PLAYER)
        game.init()
        return game

    def restart_game(self):
        self.game.new_episode()

    def get_frame(self):
        frame = self.game.get_state().screen_buffer
        return frame

    ''' Processes Doom screen image to produce cropped and resized image.'''
    def process_frame(self, frame):
        processed = frame[10:-10, 30:-30]
        processed = scipy.misc.imresize(processed, [self.width, self.width])
        # also normalize
        processed = np.reshape(processed, [np.prod(processed.shape)]) / 255.0
        return processed

    def game_finished(self):
        return self.game.is_episode_finished()

    def make_action(self, action_index):
        reward = self.env.make_action(self.actions[action_index]) / 100.0
        return reward

''' PLE Catcher game '''

class CatcherWrapper:
    def __init__(self, width, lives = 6):
        '''
            @width : width of game window
            @lives : number of deaths before the episode terminates (death = pallet does not catch ball)
        '''
        self.game = None
        self.actions = None
        self.max_game_len = 1000

        # Create game env
        catcher = Catcher(width=width, height=width,init_lives=lives)
        self.game = self.set_catcher_game_setup(catcher)


    def set_catcher_game_setup(self, game):
        p = PLE(game, display_screen=False)
        self.actions = p.getActionSet()
        p.init()
        return p

    def restart_game(self):
        self.game.reset_game()
        frame_skip = random.randint(0,30)

        #Randomize start
        for i in range(frame_skip):
            reward = self.make_action(random.choice(range(len(self.actions))))

    def get_frame(self):
        frame = self.game.getScreenGrayscale()
        color_frame = self.game.getScreenRGB()
        return frame, color_frame

    def process_frame(self, frame):
        #normalize
        processed = np.reshape(frame, [np.prod(frame.shape), 1]) / 255.0
        return processed

    def game_finished(self):
        return self.game.game_over()

    def make_action(self, action_index):
        reward = self.game.act(self.actions[action_index])
        return reward


class GameWrapper:
    def __init__(self, game_name, window_width):
        '''
            @game_name : name of required game
            @window_width : width of window/image to be used
        '''
        self.game = None

        if game_name == 'Doom':
            self.game = DoomWrapper(window_width)
            self.game.name = 'Doom'
        if game_name == 'Catcher':
            self.game = CatcherWrapper(window_width)
            self.game.name = 'Catcher'
        if game_name == 'LabMaze':
            self.game = LabWrapper(window_width)
            self.game.name = 'LabMaze'

    def get_game(self):
        return self.game

