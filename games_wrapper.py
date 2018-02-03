#from vizdoom import *
import scipy
import numpy as np
from ple.games.catcher import Catcher
from ple.games.raycastmaze import RaycastMaze
from ple.games.pixelcopter_v2 import Pixelcopter_v2
from ple import PLE
from PIL import Image
import deepmind_lab
import random
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
        self.visitation_map = {}
        self.reward_positions = []
        # Create game env
        self.game = self.set_lab_game_setup()
        # Reset game
        self.restart_game()
        self.top_down_view=None

    def construct_visitation_map(self):
        return None

    def _action(self,*entries):
        return np.array(entries, dtype=np.intc)

    def set_lab_game_setup(self):
        level = 'nav_maze_static_01'
        level = 'small_maze'
        #level = 'small_maze_multimap'

        env = deepmind_lab.Lab(
            level,
            ['RGB_INTERLACED',
             'DEBUG.POS.TRANS',
             'DEBUG.CAMERA.TOP_DOWN'],
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
        self.visitation_map[self.frames_played] = self.game.observations()[
            'DEBUG.POS.TRANS']

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

        if self.game_finished():
            self.visitation_map[self.frames_played] = self.visitation_map[
                self.frames_played - 1] + (1,0,0)
        else:
            self.visitation_map[self.frames_played] = self.game.observations()['DEBUG.POS.TRANS']
        if reward > 0 and not self.game_finished():
            self.reward_positions.append(self.game.observations()[
            'DEBUG.POS.TRANS'])



        return reward

    def construct_visitation_map(self):
        # mark rewards on map with red
        max_pos_x = 0
        max_pos_y = 0
        min_pos_x = 9999
        min_pos_y = 9999
        for position in self.visitation_map:
            (x,y,rot) = self.visitation_map[position]
            if x > max_pos_x:
                max_pos_x = x
            if x < min_pos_x:
                min_pos_x = x
            if y > max_pos_y:
                max_pos_y = y
            if y < min_pos_y:
                min_pos_y = y

        print('image of shape (%d,%d,%d)'%(483,483,4))
        print('bias x %d bias y %d'%(min_pos_x,min_pos_y))
        print('max pos x %d max pos y %d'%(max_pos_x, max_pos_y))

        image = np.ones((int(483+116), int(483+116),4))
        image.fill(255)
        image[:,:,3].fill(0)

        transparency = np.linspace(20,255,len(self.visitation_map))
        print('total steps %d '%len(self.visitation_map))

        step_color = (128,0,255)

        for timestep in range(len(self.visitation_map)):
            (x,y,z) = self.visitation_map[timestep]
            x = int(x)
            y = int(y)
            if timestep == 0 :
                #red is start point
                cv2.circle(image, (x,y), 7, (0,0,255,255), thickness=-1,
                           lineType=8, shift=0)
            elif timestep == len(self.visitation_map) - 1:
                # blue is end point
                cv2.circle(image, (x, y), 7, (255, 0 , 0,255), thickness=-1,
                           lineType=8, shift=0)
            else:
                cv2.circle(image, (x, y), 4, (128,0,128,int(transparency[
                                                                timestep])),
                           thickness=-1,
                           lineType=8, shift=0)



        return image

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
    def __init__(self, width, lives = 1):
        '''
            @width : width of game window
            @lives : number of deaths before the episode terminates (death = pallet does not catch ball)
        '''
        self.width = width
        self.game = None
        self.actions = None
        self.max_game_len = 150
        self.visitation_map = {}
        self.timer = 0
        self.coordinates = (0, 0)

        # Create game env
        catcher = Catcher(width=width, height=width,init_lives=lives)
        self.game = self.set_catcher_game_setup(catcher)


    def set_catcher_game_setup(self, game):
        p = PLE(game, display_screen=False)
        self.actions = p.getActionSet()
        p.init()
        return p

    def restart_game(self):
        self.visitation_map = {}
        self.timer = 0
        self.coordinates = (0, 0)
        self.game.reset_game()
        frame_skip = random.randint(0,30)

        #Randomize start
        for i in range(frame_skip):
            reward = self.make_action(random.choice(range(len(self.actions))))

        self.coordinates = (self.game.game.getGameState()['player_x'],
                            10)

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

        #update visitation map
        self.coordinates = (self.game.game.getGameState()['player_x'],
                            10)

        self.visitation_map[self.timer] = self.coordinates
        self.timer += 1

        return reward

    def construct_visitation_map(self):
        image = np.uint8(np.zeros((11,self.width, 4)))
        image = Image.fromarray(image)
        image = image.convert("RGBA")
        pixels = image.load()
        opacity = 100
        increase = 20


        for timestep in self.visitation_map:
            coordinate = self.visitation_map[timestep]
            if pixels[coordinate[0], coordinate[1]] == (0,0,0,0):
                pixels[coordinate[0],coordinate[1]] = (255,0,0, int(opacity))
            else:
                value = tuple(sum(x) for x in zip(pixels[coordinate[0],
                                                         coordinate[1]],
                                                  (0, 0, 0, int(increase))))
                pixels[coordinate[0], coordinate[1]] = value

        #mark start and end positions
        coordinate = self.visitation_map[0]
        pixels[coordinate[0], coordinate[1]/2] = (0,255,0,255)
        pixels[coordinate[0], coordinate[1]/2 - 1] = (0, 255, 0, 255)
        coordinate = self.visitation_map[len(self.visitation_map) - 1]
        # rewrite coordinate
        pixels[coordinate[0], coordinate[1]/2] = (0, 0, 255, 255)

        return image

#TODO GENERIC PLE WRAPPER

class RaycastMazeWrapper:
    def __init__(self, width):
        '''
            @width : width of game window
        '''
        self.game = None
        self.actions = None

        # Maximum 1000 steps in maze
        self.max_game_len = 500
        self.frames_no = 0

        # Create game env
        raycast = RaycastMaze(width=width, height=width, map_size=6)
        self.game = self.set_maze_game_setup(raycast)

    def set_maze_game_setup(self, game):
        '''
                    @game : game instance
        '''
        p = PLE(game, display_screen=False)
        #In some games, doing nothing is a valid action
        #in a maze, it is not
        self.actions = p.getActionSet()[:-1]
        p.init()
        return p

    def restart_game(self):
        self.game.reset_game()
        frame_skip = random.randint(0, 30)

        # Randomize start
        for i in range(frame_skip):
            reward = self.make_action(random.choice(range(len(self.actions))))

    def get_frame(self):
        frame = self.game.getScreenGrayscale()
        color_frame = self.game.getScreenRGB()
        return frame, color_frame

    def process_frame(self, frame):
        '''
            @frame : frame to be processed
        '''
        # normalize
        processed = np.reshape(frame, [np.prod(frame.shape), 1]) / 255.0

        return processed

    def game_finished(self):
        return self.game.game_over()

    def make_action(self, action_index):
        '''
            @action_index : index of action
        '''
        reward = self.game.act(self.actions[action_index])
        return reward


class PixelcopterWrapper:
    def __init__(self, width):
        '''
            @width : width of game window
        '''
        self.game = None
        self.actions = None

        # Maximum 1000 steps in maze
        self.max_game_len = 300
        self.frames_no = 0

        # Create game env
        raycast = Pixelcopter_v2(width=width, height=width)
        self.game = self.set_maze_game_setup(raycast)

    def construct_visitation_map(self):
        return  None

    def set_maze_game_setup(self, game):
        '''
                    @game : game instance
        '''
        p = PLE(game, display_screen=False)
        self.actions = p.getActionSet()
        p.init()
        return p

    def restart_game(self):
        self.game.reset_game()

        #don't randomize start since  it will most likely end the game
        #frame_skip = random.randint(0, 30)
        # Randomize start
        #for i in range(frame_skip):
        #    reward = self.make_action(random.choice(range(len(self.actions))))

    def get_frame(self):
        frame = self.game.getScreenGrayscale()
        color_frame = self.game.getScreenRGB()
        return frame, color_frame

    def process_frame(self, frame):
        '''
            @frame : frame to be processed
        '''
        # normalize
        processed = np.reshape(frame, [np.prod(frame.shape), 1]) / 255.0

        return processed

    def game_finished(self):
        return self.game.game_over()

    def make_action(self, action_index):
        '''
            @action_index : index of action
        '''
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
        if game_name == 'Maze':
            self.game = RaycastMazeWrapper(window_width)
            self.game.name = 'Maze'
        if game_name == 'Copter':
            self.game = PixelcopterWrapper(window_width)
            self.game.name = 'Copter'
        if game_name == 'LabMaze':
            self.game = LabWrapper(window_width)
            self.game.name = 'LabMaze'


    def get_game(self):
        return self.game

