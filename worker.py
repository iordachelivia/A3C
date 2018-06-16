from helper import *
from games_wrapper import GameWrapper
from network import NetworkWrapper
from random import shuffle
import time
#from memory_profiler import profile
import cv2
from tensorflow.contrib.tensorboard.plugins import projector
import copy

''' Worker class'''
class Worker():
    def __init__(self, index, session, trainer, base_path, global_episodes,
                 master_network, FLAGS):
        '''
            @index : worker id
            @trainer : trainer used for training the network
            @model_path : where to save the model
            @global_episodes : number of episodes => updated only by one worker
            @master_network : the master network. the workers update it with gradients
                                they also sync with the master network
            @FLAGS : different flags for the network
        '''
        self.name = "thread_" + str(index)
        self.scope = self.name
        self.index = index
        self.base_path = base_path
        self.frames_path = base_path + '/frames'
        self.session = session
        self.trainer = trainer
        self.device = FLAGS.device
        self.backup_step = FLAGS.backup_step

        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.master_network = master_network

        #Initialize A3C network for worker
        network_wrapper = NetworkWrapper(self.scope, self.trainer, self.master_network,FLAGS)
        self.network_type = FLAGS.network_type
        self.network = network_wrapper.get_network()

        self.FLAGS = FLAGS

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.sync_to_master = self.update_target_graph('global', self.name)
        self.summary_writer = tf.summary.FileWriter(base_path + "/train_" + str(index), self.session.graph)

        # This is a shameless hack and should not be here
        if self.name == 'thread_0' and self.FLAGS.has_vqvae_prediction:
            # Visualize embeddings
            config_proj = projector.ProjectorConfig()
            embedding = config_proj.embeddings.add()
            embedding.tensor_name = self.network.embeds.name
            embedding.metadata_path = os.path.join(base_path + "/train_" + str(index),'embedding_metadata.tsv')
            # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
            # read this file during startup.
            projector.visualize_embeddings(self.summary_writer, config_proj)

    ''' Synchronizes 2 networks
            It copies the first network's parameters into the second'''
    def update_target_graph(self, from_scope, to_scope):
        '''
            @from_scope : master network scope
            @to_scope : local worker network scope
        '''
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    ''' Initialize game '''
    def init_game(self, game_name, window_width):
        '''
            @game_name : The name of the selected game
            @window_width : The input image size
        '''
        gameclass = GameWrapper(game_name, window_width)
        self.game = gameclass.get_game()

    ''' Computes the discounted return and advantages'''
    def compute_discounted_return(self, values, bootstrap_value):
        '''
            @values : the values/rewards for which to compute the discounted returns
            @bootstrap_value : value to be used for bootstrapping
        '''
        #The older values are discounted more than recent ones
        values.reverse()

        discount_return = bootstrap_value
        discounted_returns = []
        #Sum over k(gamma**k * value/reward)
        for value in values:
            discount_return = value + self.network.gamma * discount_return
            discounted_returns.append(discount_return)

        discounted_returns.reverse()
        return discounted_returns

    ''' Skew distribution
        Equally sample 0-reward, positive-reward and negative reward
        (The paper samples just 3 observations) '''
    def get_auxiliary_tasks_input_reward_prediction(self, experience_replay):
        zero_valued = []
        positive_valued = []
        negative_valued = []

        for observation in experience_replay:
            [frame, selected_action, reward, new_frame, value] = observation
            if reward == 0:
                zero_valued.append(observation)
            elif reward > 0:
                positive_valued.append(observation)
            else:
                negative_valued.append(observation)

        #Suppose we always have at least one of each
        shuffle(zero_valued)
        shuffle(positive_valued)
        shuffle(negative_valued)

        #We might not have.
        all_valued = zero_valued + positive_valued + negative_valued

        if len(zero_valued) == 0:
            zero_valued = all_valued
        if len(positive_valued) == 0:
            positive_valued = all_valued
        if len(negative_valued) == 0:
            negative_valued = all_valued

        zero_val, pos_val, neg_val = zero_valued[0], positive_valued[0], negative_valued[0]
        #Sample 3 observations
        sampled_frames = [zero_val[0], pos_val[0], neg_val[0]]
        sampled_rewards = [zero_val[2], pos_val[2],neg_val[2]]
        sampled_rewards_onehot = []

        #Create onehot vectors
        for reward in sampled_rewards:
            onehot = [0, 0, 0]
            if reward == 0:
                onehot = [1, 0, 0]
            elif reward == 1:
                onehot = [0, 1, 0]
            else:
                onehot = [0, 0, 1]

            sampled_rewards_onehot.append(onehot)

        feed_dict_aux = {self.network.previous_observations_rp: sampled_frames,
                         self.network.reward_prediction_target: sampled_rewards_onehot}
        params = [self.network.rp_loss]
        return [sampled_frames, sampled_rewards_onehot], params, feed_dict_aux

    ''' Do not skew distribution
            Sample BACKUP_STEP consecutive observations
    '''
    def get_auxiliary_tasks_input_pixel_control(self, experience_replay,
                                                session):
        index = np.random.randint(0, len(experience_replay) - self.backup_step - 1)
        sequence = experience_replay[index: index + self.backup_step]

        # observation = [frame, selected_action, reward, new_frame, value]
        # Get frames
        frames = [x[0] for x in sequence]
        rewards = [x[2] for x in sequence]
        actions = [x[1] for x in sequence]

        # We have to pass the last frame through the network to get the bootstrap value
        feed_dict = {
            self.network.previous_observations_pc: [frames[len(frames) - 1]]}
        value_bootstrap = session.run(self.network.pc_q_max, feed_dict)[0]

        # Compute discounted returns for value prediction loss
        discounted_values = self.compute_discounted_return(list(rewards),
                                                           value_bootstrap)
        discounted_values = np.asarray(discounted_values)

        #one hot the actions
        actions_one_hot = np.zeros(([len(actions),self.network.action_size]))
        actions_one_hot[np.arange(len(actions)), actions] = 1
        feed_dict_aux = {self.network.previous_observations_pc: frames,
                         self.network.pc_actions_taken: actions_one_hot,
                         self.network.q_target: discounted_values}
        params = [self.network.pc_loss]

        return [frames, discounted_values], params, feed_dict_aux

    ''' Do not skew distribution
                Sample BACKUP_STEP consecutive observations
        '''

    def get_auxiliary_tasks_input_frame_prediction(self, experience_replay,
                                                session):
        k_frames = 1
        index = np.random.randint(0,
                                  len(experience_replay) - self.backup_step -
                                  k_frames)
        sequence = experience_replay[index: index + self.backup_step]
        sequence_target = experience_replay[index + k_frames: index +
                                                       self.backup_step + k_frames]
        # observation = [frame, selected_action, reward, new_frame, value]
        # Get frames
        frames = [x[0] for x in sequence]
        frames_target = [x[0] for x in sequence_target]

        
        actions = [x[1] for x in sequence]

        feed_dict_aux = {self.network.previous_observations_fp: frames,
                         self.network.fp_previous_frames_target: frames_target}

        if self.FLAGS.concat_action_lstm:
            feed_dict_aux2 = {
                self.network.previous_observations_fp_actions: actions
            }
            feed_dict_aux.update(feed_dict_aux2)

        params = [self.network.fp_loss,
                  self.network.fp_previous_frames_target,
                  self.network.fp_reconstruction]

        return [frames], params, feed_dict_aux

    def get_auxiliary_tasks_input_frame_prediction_thresholded(self,
                                                         experience_replay,
                                                   session):
        k_frames = 1
        index = np.random.randint(0,
                                  len(experience_replay) - self.backup_step -
                                  k_frames)
        sequence = experience_replay[index: index + self.backup_step]
        sequence_target = experience_replay[index + k_frames: index +
                                                              self.backup_step + k_frames]
        # observation = [frame, selected_action, reward, new_frame, value]
        # Get frames
        frames = [x[0] for x in sequence]
        frames_target = [x[0] for x in sequence_target]

        # threshold target frames so they are black and white
        frames_target_thresholded = []
        for frame in frames_target:
            frame = frame * 255

            reshaped = np.reshape(frame, (84,84))
            reshaped = reshaped.astype(np.uint8)
            median = cv2.medianBlur(reshaped, 15)
            ret, thresh = cv2.threshold(median, 127, 255, 0)
            thresh = thresh / 255.0
            reshaped = np.reshape(thresh,frame.shape)
            frames_target_thresholded.append(reshaped)


        actions = [x[1] for x in sequence]

        feed_dict_aux = {self.network.previous_observations_fp_thresh: frames,
                         self.network.current_observations_fp_thresh:frames_target,
                         self.network.fp_thresh_previous_frames_target:frames_target_thresholded }


        params = [self.network.fp_thresh_loss,
                  self.network.current_observations_fp_thresh,
                  self.network.fp_thresh_previous_frames_target,
                  self.network.fp_thresh_reconstruction_vis ]

        return [frames], params, feed_dict_aux


    ''' Do not skew distribution
                Sample BACKUP_STEP consecutive observations
        '''
    def convert_flow_to_bgr(self, flow):
        hsv = np.zeros((84,84,3),dtype=np.uint8)
        hsv[..., 1] = 255
        flow = np.reshape(flow,(84,84,2))
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

    def get_auxiliary_tasks_input_flow_prediction(self, experience_replay,
                                                session):
        k_frames = 1
        index = np.random.randint(0,
                                  len(experience_replay) - self.backup_step -
                                  k_frames)
        sequence = experience_replay[index: index + self.backup_step]
        sequence_target = experience_replay[index + k_frames: index +
                                                       self.backup_step + k_frames]
        # observation = [frame, selected_action, reward, new_frame, value]
        # Get frames
        frames = [x[0] for x in sequence]
        frames_target = [x[0] for x in sequence_target]

        # compute flow from frames and target frames
        flow_target = []
        for (prvs,nxt) in zip(frames, frames_target):
            # un-normalize images
            prvs = (prvs * 255.).astype(np.uint8)
            prvs = np.reshape(prvs,(84,84))
            nxt = (nxt * 255.).astype(np.uint8)
            nxt = np.reshape(nxt, (84, 84))

            flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15,
                                                3, 5,
                                            1.2, 0)


            flow_target.append(np.reshape(flow,newshape=(-1,2)))

        feed_dict_aux = {self.network.previous_observations_fl: frames,
                         self.network.previous_observations_fl_next :
                             frames_target,
                         self.network.fl_target: flow_target}

        params = [self.network.fl_loss,
                  self.network.previous_observations_fl,
                  self.network.previous_observations_fl_next,
                  self.network.fl_target,
                  self.network.flow_prediction]

        return [frames], params, feed_dict_aux

    ''' Do not skew distribution
                Sample BACKUP_STEP consecutive observations
        '''

    def get_auxiliary_tasks_input_vqvae_prediction(self, experience_replay,
                                                   session):
        k_frames = 0
        index = np.random.randint(0,
                                  len(experience_replay) - self.backup_step -
                                  k_frames)
        sequence = experience_replay[index: index + self.backup_step]
        sequence_target = experience_replay[index + k_frames: index +
                                                              self.backup_step + k_frames]
        # observation = [frame, selected_action, reward, new_frame, value]
        # Get frames
        frames = [x[0] for x in sequence]
        frames_target = [x[0] for x in sequence_target]

        feed_dict_aux = {self.network.previous_observations_vqvae: frames,
                         self.network.vqvae_target: frames_target}

        params = [self.network.vqvae_loss, self.network.vqvae_indices_ids,
                  self.network.vqvae_indices_ids_count,
                  self.network.vqvae_target,
                  self.network.vqvae_prediction]

        return [frames], params, feed_dict_aux

    ''' Do not skew distribution
                Sample BACKUP_STEP consecutive observations
        '''

    def get_auxiliary_tasks_input_action_prediction(self, experience_replay,
                                                session):
        index = np.random.randint(0,
                                  len(experience_replay) - self.backup_step - 1)
        sequence = experience_replay[index: index + self.backup_step]

        # observation = [frame, selected_action, reward, new_frame, value]
        # Get frames
        frames = [x[0] for x in sequence]
        frames_next = [x[3] for x in sequence]
        action_target = [x[1] for x in sequence]

        lstm_state = self.network.lstm_state_init

        feed_dict_aux = {self.network.first_state_ap: frames,
                         self.network.second_state_ap: frames_next,
                         self.network.action_prediction_target:action_target,
                         self.network.initial_cell_c_state_ap: lstm_state[0],
                         self.network.initial_cell_h_state_ap: lstm_state[1]
                         }
        params = [self.network.ap_loss]

        return [frames, frames_next, action_target], params, feed_dict_aux

    ''' Do not skew distribution
        Sample BACKUP_STEP consecutive observations
    '''
    def get_auxiliary_tasks_input_value_prediction(self, experience_replay, session):
        index = np.random.randint(0, len(experience_replay) - self.backup_step - 1)
        sequence = experience_replay[index: index + self.backup_step]

        #observation = [frame, selected_action, reward, new_frame, value]
        #Get frames
        frames = [x[0] for x in sequence]
        rewards = [x[2] for x in sequence]
        actions = [x[1] for x in sequence]

        #We have to pass the last frame through the network to get the bootstrap value
        feed_dict = {self.network.previous_observations_vp:
                         [frames[len(frames)-1]]}

        if self.FLAGS.concat_action_lstm:
            feed_dict_aux = {
                self.network.previous_observations_vp_actions:
                    [actions[len(actions) - 1]]
            }
            feed_dict.update(feed_dict_aux)

        value_bootstrap = session.run(self.network.vp_prediction, feed_dict)[0]

        #Compute discounted returns for value prediction loss
        discounted_values = self.compute_discounted_return(list(rewards), value_bootstrap)
        discounted_values = np.asarray(discounted_values)

        feed_dict_aux = {self.network.previous_observations_vp : frames,
                        self.network.value_prediction_target: discounted_values}

        if self.FLAGS.concat_action_lstm:
            feed_dict_aux2 = {
                self.network.previous_observations_vp_actions: actions
            }
            feed_dict_aux.update(feed_dict_aux2)

        params = [self.network.vp_loss]

        return [frames, discounted_values], params, feed_dict_aux

    ''' Create input for auxiliary tasks '''
    def get_auxiliary_tasks_input(self, experience_replay, session):
        input_data = []
        params = []
        feed_dict = {}

        #TODO DO NOT MODIFY ORDER!!!
        #IN FUNCTION BACK IT UP
        if self.FLAGS.has_reward_prediction :
            input_data_aux, params_aux, feed_dict_aux = \
                self.get_auxiliary_tasks_input_reward_prediction(experience_replay)
            input_data.extend(input_data_aux)
            params.extend(params_aux)
            feed_dict.update(feed_dict_aux)
        if self.FLAGS.has_pixel_control :
            input_data_aux, params_aux, feed_dict_aux = \
                self.get_auxiliary_tasks_input_pixel_control(
                    experience_replay,session)
            input_data.extend(input_data_aux)
            params.extend(params_aux)
            feed_dict.update(feed_dict_aux)
        if self.FLAGS.has_value_prediction :
            input_data_aux, params_aux, feed_dict_aux = \
                self.get_auxiliary_tasks_input_value_prediction(experience_replay, session)
            input_data.extend(input_data_aux)
            params.extend(params_aux)
            feed_dict.update(feed_dict_aux)
        if self.FLAGS.has_frame_prediction :
            input_data_aux, params_aux, feed_dict_aux = \
                self.get_auxiliary_tasks_input_frame_prediction(experience_replay,
                                                       session)
            input_data.extend(input_data_aux)
            params.extend(params_aux)
            feed_dict.update(feed_dict_aux)
        if self.FLAGS.has_action_prediction :
            input_data_aux, params_aux, feed_dict_aux = \
                self.get_auxiliary_tasks_input_action_prediction(
                    experience_replay, session)
            input_data.extend(input_data_aux)
            params.extend(params_aux)
            feed_dict.update(feed_dict_aux)
        if self.FLAGS.has_flow_prediction:
            input_data_aux, params_aux, feed_dict_aux = \
                self.get_auxiliary_tasks_input_flow_prediction(
                    experience_replay, session)
            input_data.extend(input_data_aux)
            params.extend(params_aux)
            feed_dict.update(feed_dict_aux)
        if self.FLAGS.has_vqvae_prediction:
            input_data_aux, params_aux, feed_dict_aux = \
                self.get_auxiliary_tasks_input_vqvae_prediction(
                    experience_replay, session)
            input_data.extend(input_data_aux)
            params.extend(params_aux)
            feed_dict.update(feed_dict_aux)
        if self.FLAGS.has_frame_prediction_thresholded:
            input_data_aux, params_aux, feed_dict_aux = \
                self.get_auxiliary_tasks_input_frame_prediction_thresholded(
                    experience_replay, session)
            input_data.extend(input_data_aux)
            params.extend(params_aux)
            feed_dict.update(feed_dict_aux)

        return input_data, params, feed_dict

	
    ''' Compute gradients and update master network '''
    def back_it_up(self, episode_buffer, experience_replay, bootstrap_value, session):
        '''
            @episode_buffer : buffer for the episode
            @bootstrap_value : value to be used for bootstrapping in computing the discounted returns
            @session : the current tensorflow session to be used
        '''
        # Extract data from episode buffer
        rollout = np.array(episode_buffer)
        frames = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_frames = rollout[:, 3]
        values = rollout[:, 4]

        # Compute discounted returns and advantages -> generalized advantage estimation
        #Discounted rewards = discounted(rewards)
        discounted_returns = self.compute_discounted_return(list(rewards), bootstrap_value)
        discounted_returns = np.asarray(discounted_returns)
        td = self.network.gamma * np.asarray(values.tolist() + [bootstrap_value])[1:] - values
        advantages = rewards + td
        #Generalized advantage estimation = discounted(rewards + gamma * values_t+1 - values_t)
        advantages = self.compute_discounted_return(list(advantages), 0)
        advantages = np.asarray(advantages)

        # Get parameters for backup operation
        feed_dict = {self.network.input: np.stack(frames),
                     self.network.actions: actions,
                     self.network.discounted_rewards: discounted_returns,
                     self.network.advantages: advantages,
                     }

        params = [self.network.value_loss, self.network.policy_loss, self.network.entropy,
                  self.network.clipped_grad_norm, self.network.apply_grads]

        if self.network_type == 'LSTM':
            # We reset the lstm state
            lstm_state = self.network.lstm_state_init

            feed_dict_lstm = {self.network.initial_cell_c_state: lstm_state[0],
                              self.network.initial_cell_h_state: lstm_state[1]}
            feed_dict.update(feed_dict_lstm)

            if self.FLAGS.concat_action_lstm:
                feed_dict_aux = {
                    self.network.previous_observations_actions : actions
                }
                feed_dict.update(feed_dict_aux)

        #Auxiliary tasks additional input
        #Only auxiliary tasks use experience replay
        input_data, params_aux, feed_dict_aux = self.get_auxiliary_tasks_input(experience_replay, session)
        params.extend(params_aux)
        feed_dict.update(feed_dict_aux)

        # Backup
        params.append(self.network.loss)
        results = session.run(params,feed_dict)
        value_loss, policy_loss, entropy, clipped_grad_norm = results[:4]

        start_index = 5
        reward_prediction_loss = None
        pixel_control_loss = None
        value_prediction_loss = None
        frame_prediction_loss = None
        frame_reconstruction = None
        frame_target = None

        frame_threshold_prediction_loss = None
        frame_threshold_orig_image = None
        frame_threshold_target = None
        frame_threshold_prediction = None

        action_prediction_loss = None
        flow_prediction_loss = None
        flow_input = None
        flow_input_next = None
        flow_target = None
        flow_prediction = None
        vqvae_prediction_loss = None
        vqvae_indices_ids = None
        vqvae_indices_ids_count = None
        vqvae_target = None
        vqvae_prediction = None

        if self.FLAGS.has_reward_prediction:
            reward_prediction_loss = results[start_index] /len(rollout)
            start_index += 1
        if self.FLAGS.has_pixel_control:
            pixel_control_loss = results[start_index] /len(rollout)
            start_index += 1
        if self.FLAGS.has_value_prediction:
            value_prediction_loss = results[start_index] /len(rollout)
            start_index += 1
        if self.FLAGS.has_frame_prediction:
            frame_prediction_loss = results[start_index]/len(rollout)
            start_index += 1
            frame_target = results[start_index]
            start_index += 1
            frame_reconstruction = results[start_index]
            start_index += 1
        if self.FLAGS.has_action_prediction:
            action_prediction_loss = results[start_index]/len(rollout)
            start_index += 1
        if self.FLAGS.has_flow_prediction:
            flow_prediction_loss = results[start_index]/len(rollout)
            start_index += 1
            flow_input = results[start_index]
            start_index += 1
            flow_input_next = results[start_index]
            start_index += 1
            flow_target = results[start_index]
            start_index += 1
            flow_prediction = results[start_index]
            start_index += 1
        if self.FLAGS.has_vqvae_prediction:
            vqvae_prediction_loss = results[start_index] / len(rollout)
            start_index += 1
            vqvae_indices_ids = results[start_index]
            start_index += 1
            vqvae_indices_ids_count = results[start_index]
            start_index += 1
            vqvae_target = results[start_index]
            start_index += 1
            vqvae_prediction = results[start_index]
            start_index += 1
        if self.FLAGS.has_frame_prediction_thresholded:
            frame_threshold_prediction_loss = results[start_index]
            start_index += 1
            frame_threshold_orig_image = results[start_index]
            start_index += 1
            frame_threshold_target = results[start_index]
            start_index += 1
            frame_threshold_prediction = results[start_index]
            start_index += 1

        #network loss
        total_loss = results[-1] /len(rollout)

        # Save values to plot later
        values_to_plot = [value_loss / len(rollout), policy_loss / len(rollout), entropy / len(rollout),
                               clipped_grad_norm, reward_prediction_loss,
                          pixel_control_loss, value_prediction_loss,
                          frame_prediction_loss, frame_target,
                          frame_reconstruction,
                          action_prediction_loss,
                          flow_prediction_loss, flow_input, flow_input_next,
                          flow_target,
                          flow_prediction,
                          vqvae_prediction_loss,
                          vqvae_indices_ids, vqvae_indices_ids_count,
                          vqvae_target, vqvae_prediction,
                          frame_threshold_prediction_loss,
                          frame_threshold_orig_image, frame_threshold_target,
                          frame_threshold_prediction,
                          total_loss]

        return values_to_plot

    def write_mean_summary(self, mean_reward, mean_positive_reward,
                           mean_negative_reward, mean_len, mean_value,
                           episode_count):
        summary = tf.Summary();
        summary.value.add(tag='Performance/Mean rewards', simple_value=float(mean_reward))
        summary.value.add(tag='Performance/Mean positive rewards', simple_value=float(mean_positive_reward))
        summary.value.add(tag='Performance/Mean negative rewards', simple_value=float(mean_negative_reward))
        summary.value.add(tag='Performance/Mean episode lengths', simple_value=float(mean_len))
        summary.value.add(tag='Performance/Mean value', simple_value=float(mean_value))
        self.summary_writer.add_summary(summary, episode_count)
        self.summary_writer.flush()

    ''' Write a summary '''

    def write_summary(self, episode_count, total_frames, frames, losses, session, saver):
        '''
            @episode_count : x-axis value
            @total_frames : the number of frames seen
            @frames : episode frames
            @mean_reward : mean reward gathered
            @mean_len : mean episode length
            @mean_value : mean values computed
            @losses : different losses to be plotted
            @session : the current tensorflow session
            @saver : used to save the model
        '''

        #Extract losses
        [value_loss, policy_loss, entropy, clipped_grad_norm, rp_loss,
         pc_loss, vp_loss, fp_loss, fp_target, fp_recon, ap_loss, fl_loss,
         fl_input, fl_input_next,
         fl_target, fl_pred, vqvae_loss, vqvae_indices_ids,
         vqvae_indices_ids_count, vqvae_target, vqvae_prediction,
         fp_thresh_loss, fp_thresh_image, fp_thresh_target,
         fp_thresh_prediction,
         total_loss] = \
            losses

        #Create gifs from frames

        if self.scope == 'thread_0':
            if (episode_count <=10 and episode_count % 5 == 0) or \
                                    episode_count % 15 == 0:
                images = np.array(frames)
                time_per_step = 0.05
                make_gif(images, self.frames_path+'/image' + str(episode_count) + '.avi',
                         duration=len(images) * time_per_step, true_image=True, salience=False)

                #save visitation map
                visitation_map= self.game.construct_visitation_map()
                cv2.imwrite(
                         self.frames_path+'/visitation_map_' +
                                        str(episode_count) + '.png', visitation_map)
                #exit(1)

        #Save model at each 250 episodes
        if episode_count % 25 == 0 and self.name == 'thread_0':
            #saver.save(session, os.path.join(self.model_path, 'model.ckpt'), self.global_episodes)
            path = self.base_path + "/train_" + str(self.index)
            saver.save(session, os.path.join(path, 'model.ckpt'), self.global_episodes)
            print ("LOG: Saved Model")

        #Plot values
        summary = tf.Summary();
        summary.value.add(tag='Losses/Value loss', simple_value=float(value_loss))
        summary.value.add(tag='Losses/Policy Loss', simple_value=float(policy_loss))
        summary.value.add(tag='Losses/Entropy', simple_value=float(entropy))
        summary.value.add(tag='Losses/Total Loss', simple_value=float(total_loss))
        summary.value.add(tag='Losses/Clipped Grad Norm', simple_value=float(clipped_grad_norm))
        summary.value.add(tag='Performance/Total frames seen', simple_value=int(total_frames))

        if rp_loss != None:
            summary.value.add(tag='Losses/AuxLosses/Reward Prediction',
                              simple_value=float(rp_loss))
        if pc_loss != None:
            summary.value.add(tag='Losses/AuxLosses/Pixel Control',
                              simple_value=float(pc_loss))
        if vp_loss != None:
            summary.value.add(tag='Losses/AuxLosses/Value Prediction',
                              simple_value=float(vp_loss))
        if fp_loss != None:
            summary.value.add(tag='Losses/AuxLosses/Frame Prediction',
                              simple_value=float(fp_loss))

            if self.scope == 'thread_0':
                #normalize frame
                fp_recon = fp_recon[0] * 255
                fp_recon = np.reshape(fp_recon, (84,84))
                fp_recon = np.array(fp_recon, dtype=np.uint8)
                cv2.imwrite(os.path.join(self.frames_path,
                                         str(self.scope)+'_fp_reconstruction_'
                                         + str(
                    episode_count) + '.jpg'),fp_recon)

                fp_target = fp_target[0] * 255
                fp_target = np.reshape(fp_target, (84, 84))
                fp_target = np.array(fp_target, dtype=np.uint8)
                cv2.imwrite(os.path.join(self.frames_path,
                                         str(self.scope) + '_fp_target_' + str(
                                             episode_count) + '.jpg'),
                            fp_target)

        if ap_loss != None:
            summary.value.add(tag='Losses/AuxLosses/Action Prediction',
                                  simple_value=float(ap_loss))
        if fl_loss != None:
            summary.value.add(tag='Losses/AuxLosses/Flow Prediction',
                              simple_value=float(fl_loss))

            if self.scope == 'thread_0':
                #normalize frame
                fl_pred = fl_pred[0]
                fl_pred_rgb = self.convert_flow_to_bgr(fl_pred)
                cv2.imwrite(os.path.join(self.frames_path,
                                         str(self.scope)+'_flow_pred_' + str(
                    episode_count) + '.jpg'),fl_pred_rgb)

                fl_target = fl_target[0]
                fl_target_rgb = self.convert_flow_to_bgr(fl_target)
                cv2.imwrite(os.path.join(self.frames_path,
                                         str(self.scope) + '_flow_target_' +
                                         str(
                                             episode_count) + '.jpg'),
                            fl_target_rgb)

                # normalize frame
                fl_input = fl_input[0] * 255
                fl_input = np.reshape(fl_input, (84, 84))
                fl_input = np.array(fl_input, dtype=np.uint8)
                cv2.imwrite(os.path.join(self.frames_path,
                                         str(self.scope) +
                                         '_fl_input_a_'
                                         + str(
                                             episode_count) + '.jpg'), fl_input)

                # normalize frame
                fl_input_next = fl_input_next[0] * 255
                fl_input_next = np.reshape(fl_input_next, (84, 84))
                fl_input_next = np.array(fl_input_next, dtype=np.uint8)
                cv2.imwrite(os.path.join(self.frames_path,
                                         str(self.scope) +
                                         '_fl_input_b_'
                                         + str(
                                             episode_count) + '.jpg'), fl_input_next)

        if vqvae_loss != None:
            summary.value.add(tag='Losses/AuxLosses/VQVAE Prediction',
                              simple_value=float(vqvae_loss))
            for idx, id_name in enumerate(vqvae_indices_ids):
                summary.value.add(tag='VQVAE/Indices/'+str(id_name),
                                  simple_value=int(vqvae_indices_ids_count[idx]))

            if self.scope == 'thread_0':
                #normalize frame
                vqvae_prediction = vqvae_prediction[0] * 255
                vqvae_prediction = np.reshape(vqvae_prediction, (84,84))
                vqvae_prediction = np.array(vqvae_prediction, dtype=np.uint8)
                cv2.imwrite(os.path.join(self.frames_path,
                                         str(self.scope)+
                                         '_vqvae_reconstruction_'
                                         + str(episode_count) + '.jpg'),
                            vqvae_prediction)

                vqvae_target = vqvae_target[0] * 255
                vqvae_target = np.reshape(vqvae_target, (84, 84))
                vqvae_target = np.array(vqvae_target, dtype=np.uint8)
                cv2.imwrite(os.path.join(self.frames_path,
                                         str(self.scope) + '_vqvae_target_' +
                                         str(episode_count) + '.jpg'),
                            vqvae_target)

        if fp_thresh_loss != None:
            summary.value.add(tag='Losses/AuxLosses/Frame Prediction '
                                  'Thresholded',
                              simple_value=float(fp_thresh_loss))

            if self.scope == 'thread_0':
                #normalize frame
                fp_thresh_prediction = fp_thresh_prediction[0] * 255
                fp_thresh_prediction = np.reshape(fp_thresh_prediction, (84,84))
                fp_thresh_prediction = np.array(fp_thresh_prediction, dtype=np.uint8)
                cv2.imwrite(os.path.join(self.frames_path,
                                         str(
                                             self.scope)+'_fp_thresh_prediction_'
                                         + str(
                    episode_count) + '.jpg'),fp_thresh_prediction)

                fp_thresh_image = fp_thresh_image[0] * 255
                fp_thresh_image = np.reshape(fp_thresh_image, (84, 84))
                fp_thresh_image = np.array(fp_thresh_image, dtype=np.uint8)
                cv2.imwrite(os.path.join(self.frames_path,
                                         str(self.scope) + '_fp_thresh_image_' + str(
                                             episode_count) + '.jpg'),
                            fp_thresh_image)

                fp_thresh_target = fp_thresh_target[0] * 255
                fp_thresh_target = np.reshape(fp_thresh_target, (84, 84))
                fp_thresh_target = np.array(fp_thresh_target, dtype=np.uint8)
                cv2.imwrite(os.path.join(self.frames_path,
                                         str(
                                             self.scope) + '_fp_thresh_target_' + str(
                                             episode_count) + '.jpg'),
                            fp_thresh_target)


        self.summary_writer.add_summary(summary, episode_count)
        self.summary_writer.flush()

    ''' Experience replay as a buffer that keeps the most recent 2k entries '''
    def update_experience_replay(self, experience_replay, observation):
        experience_replay.append(observation)
        if len(experience_replay) > self.FLAGS.experience_buffer_maxlen:
            experience_replay = experience_replay[-self.FLAGS.experience_buffer_maxlen:]

        return experience_replay

	
    ''' Play an episode  '''
    def play(self, sess, experience_replay):
        '''
            @sess : current tensorflow session
        '''
        episode_buffer = []
        episode_values = []
        episode_frames = []
        episode_reward = 0
        episode_rewards_list = []
        episode_step_count = 0
        selected_action = 0
        toplot = [0, 0, 0, 0, None, None, None, None, None, None, None, None,
                  None, None, None, None, None, None,
                  None, None, None, None, None, None, None, 0]
        # Clean slate for each episode
        self.game.restart_game()
        rnn_state = None

        if self.network_type == 'LSTM':
            rnn_state = self.network.lstm_state_init

        while not self.game.game_finished():
            #Get current frame from game
            frame, colour_frame = self.game.get_frame()
            episode_frames.append(colour_frame)
            frame = self.game.process_frame(frame)

            # Take an action using probabilities from policy network output.
            feed_dict = {self.network.input: [frame]}

            if self.network_type == 'LSTM':
                feed_dict_lstm ={self.network.lstm_state[0]: rnn_state[0],
                                 self.network.lstm_state[1]: rnn_state[1]}
                feed_dict.update(feed_dict_lstm)

                if self.FLAGS.concat_action_lstm:
                    feed_dict_lstm = {
                        self.network.previous_observations_actions:
                            [selected_action]}
                    feed_dict.update(feed_dict_lstm)

                policy, value, rnn_state = sess.run([self.network.policy, self.network.value,
                                                     self.network.lstm_state_out], feed_dict)
            else:
                policy, value = sess.run([self.network.policy, self.network.value], feed_dict)

            value = value[0]
            policy = policy[0]

            #Random tie break
            try:
                selected_action = np.random.choice(np.flatnonzero(policy == policy.max()))
            except ValueError:
                print('LOG: wat')
                exit(1)

            # Make action and get associated reward
            reward_preclip = self.game.make_action(selected_action)
            # Clip rewards to -1, 1 interval
            # If maze, goal is more important than apple
            reward = np.clip(reward_preclip, -1, 1)

            episode_rewards_list.append(reward)

            game_over = self.game.game_finished()
            if not game_over :
                new_frame, colour_frame = self.game.get_frame()
                episode_frames.append(colour_frame)
                new_frame = self.game.process_frame(new_frame)
            else:
                new_frame = frame

            observation = [frame, selected_action, reward, new_frame, value]
            episode_buffer.append(observation)
            experience_replay = self.update_experience_replay(experience_replay, observation)

            episode_values.append(value)

            # Sum over received rewards
            episode_reward += reward

            frame = new_frame
            episode_step_count += 1

            if game_over:
                print('LOG: game over in the middle')
                break

            #While the experience buffer gets full, do not update
            if len(experience_replay) < self.FLAGS.experience_buffer_maxlen:
                print('LOG : %s ****** Filling experience replay buffer'%self.scope)
                continue

            # If the episode hasn't ended, but the experience buffer is full, then we
            # make an update step using that experience rollout.
            # We need to bootstrap if we cut off episode
            if len(episode_buffer) == self.backup_step or episode_step_count == self.game.max_game_len:
                # Since we don't know what the true final return is, we "bootstrap" from our current
                # value estimation.
                feed_dict = {self.network.input: [frame]}

                if self.network_type == 'LSTM':
                    feed_dict_lstm ={self.network.lstm_state[0]: rnn_state[0],
                                     self.network.lstm_state[1]: rnn_state[1]}
                    feed_dict.update(feed_dict_lstm)
                    if self.FLAGS.concat_action_lstm:
                        feed_dict_lstm = {
                            self.network.previous_observations_actions:
                                [selected_action]}
                        feed_dict.update(feed_dict_lstm)

                value_bootstrap = sess.run(self.network.value, feed_dict)[0]

                toplot = self.back_it_up(episode_buffer, experience_replay, value_bootstrap, sess)
                
                episode_buffer = []
                #We sync to the master network
                sess.run(self.sync_to_master)

            #We need to bootstrap if we cut off episode
            if episode_step_count >= self.game.max_game_len:
                break

        # Update the network using the experience buffer at the end of the episode.
        # Dont update until you have filled the experience buffer
        if len(episode_buffer) != 0 and len(experience_replay) == \
                self.FLAGS.experience_buffer_maxlen:
            toplot = self.back_it_up(episode_buffer, experience_replay, 0, sess)

        value_mean = np.mean(episode_values)
        episode_rewards_list = np.array(episode_rewards_list)

        return episode_reward, episode_rewards_list, value_mean, episode_step_count, \
               episode_frames, toplot, experience_replay

	
    ''' Worker main logic'''
    def work(self, sess, coord, saver):
        '''
            @sess : current tensorflow session
            @coord : thread coordinator
            @saver : saver for model saving
        '''
        episode_count = sess.run(self.global_episodes)
        total_frames = 0
        games_rewards = []
        positive_rewards = []
        negative_rewards = []
        experience_replay = []
        games_value_means = []
        games_lengths = []

        print ("LOG: Starting worker " + str(self.index))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                # Sync to master network
                sess.run(self.sync_to_master)

                # Play an episode
                start = time.time()
                episode_reward, episode_rewards_list, value_mean, episode_step_count, episode_frames, \
                                                        toplot, experience_replay = \
                                                            self.play(sess, experience_replay)
                end = time.time()
                print('LOG: worker %s got reward %f in %d frames in %f s '
                      'episode'%(
                    self.name, episode_reward, len(episode_frames), end-start))

                # Accumulate over 5 episodes to plot
                games_rewards.append(episode_reward)
                positive_rewards.append(episode_rewards_list[episode_rewards_list >= 0].sum())
                negative_rewards.append(episode_rewards_list[episode_rewards_list < 0].sum())
                games_value_means.append(value_mean)
                games_lengths.append(episode_step_count)
                total_frames += len(episode_frames)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if True:
                    # Write summary
                    self.write_summary(episode_count, total_frames,
                                       episode_frames, toplot, sess, saver)

                #Mean reward summary every 5 episodes
                if episode_count % 5 == 0 and episode_count != 0:
                    # Compute mean metrics to plot
                    mean_reward = np.mean(games_rewards)
                    mean_positive_reward = np.mean(positive_rewards)

                    mean_negative_reward = np.mean(negative_rewards)
                    mean_len = np.mean(games_lengths)
                    mean_value = np.mean(games_value_means)

                    self.write_mean_summary(mean_reward, mean_positive_reward,
                                       mean_negative_reward, mean_len,
                                       mean_value, episode_count)

                    # Reset buffers
                    games_rewards = []
                    negative_rewards = []
                    positive_rewards = []

                    games_value_means = []
                    games_lengths = []

                # Only one thread increments global_episodes
                if self.name == 'thread_0':
                    sess.run(self.increment)
                    global_ep = sess.run(self.global_episodes)
                    if global_ep == self.FLAGS.episodes:
                        #saver.save(sess, os.path.join(self.model_path,'model.ckpt'), global_ep)
                        path = self.base_path + "/train_" + str(self.index)
                        saver.save(sess, os.path.join(path, 'model.ckpt'), self.global_episodes)
                        print ("LOG: Saved Model")

                        coord.request_stop()
                    print('LOG: ----------------Global episode %d'%global_ep)
                episode_count += 1

                print('LOG: Worker ' + str(self.scope) + ' episode ' + str(
                    episode_count))

