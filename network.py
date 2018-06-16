from helper import *

''' Base network to be used with additional LSTM or FF network'''
class BaseNetwork:
    def __init__(self, scope, trainer,master_network, FLAGS):
        '''
            @scope : scope in which to create variables
            @trainer : trainer used to minimize loss
            @master_network : master network
            @FLAGS : specify different options through flags
        '''
        self.input_size = FLAGS.input_size
        self.no_frames = FLAGS.no_frames
        self.action_size = FLAGS.action_size
        self.scope = scope
        self.beta = FLAGS.beta
        self.gamma = FLAGS.gamma
        self.trainer = trainer
        self.master_network = master_network
        self.device = FLAGS.device
        self.FLAGS = FLAGS

        # LSTM size dependent on concat on/off
        if self.FLAGS.concat_action_lstm:
            self.lstm_additional_field = self.action_size
        else:
            self.lstm_additional_field = 0


        #Construct network in current scope
        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            #Input
            self.input = tf.placeholder(dtype=tf.float32,shape=[None, self.input_size * self.input_size, self.no_frames],name='input')
            #Reshape input to be a 4d tensor
            self.input_reshaped = tf.reshape(self.input,shape=[-1, self.input_size, self.input_size, self.no_frames])

        #We need to reuse the CNN layers for auxiliary tasks
        #So the conv layers will take an arbitrary size input parameter
        self.cnn_output = self.cnn_trunk(self.input_reshaped)

        #Fully connected 256 layer
        self.fc_output = self.fc_trunk(self.cnn_output)

    def normalized_columns_initializer(std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)

        return _initializer

    ''' Initialize weights using Xavier initialization'''
    def init_weights_and_biases(self, name, shape):
        # TODO MODIFY THIS
        if "policy" or "value" not in name:
            if len(shape) == 4:
                d = 1.0 / np.sqrt(shape[0] * shape[1] * shape[3])
            else:
                d = 1.0 / np.sqrt(shape[1])

            weights = tf.get_variable(name=name, shape=shape,
                                      initializer=tf.random_uniform_initializer(minval=-d, maxval=d))
        else:
            if "policy" in name:
                weights = tf.get_variable(name=name, shape=shape,
                                          initializer=self.normalized_columns_initializer(0.01))
            else:
                weights = tf.get_variable(name=name, shape=shape,
                                          initializer=self.normalized_columns_initializer(1.0))

        if "deconv" in name:
            biases = tf.get_variable(name='b_' + name, shape=shape[-2:-1], initializer=tf.zeros_initializer())
        else:
            biases = tf.get_variable(name='b_' + name, shape=shape[-1], initializer=tf.zeros_initializer())

        return weights, biases

    ''' Reuse first FC layer'''
    def fc_trunk(self, input, reuse=False):
        with tf.device(self.device), tf.variable_scope(self.scope, reuse=reuse) as scope:
            # FC1
            # Relu2 outputs 9x9x32
            shape = [9 * 9 * 32, 256]
            self.W_fc1, self.b_fc1 = self.init_weights_and_biases("W_fc1", shape)

            # Relu2 output needs to be reshaped
            relu2_reshaped = tf.reshape(input, [-1, 9 * 9 * 32])
            self.fc1 = tf.matmul(relu2_reshaped, self.W_fc1) + self.b_fc1

            # RELU
            relu3 = tf.nn.relu(self.fc1)

            return relu3

    ''' Reuse cnn layers for auxiliary tasks'''
    def cnn_trunk(self, input, reuse=False):
        with tf.device(self.device), tf.variable_scope(self.scope, reuse=reuse) as scope:
            # Conv1
            # Create weights and biases
            # filters 8x8x16 stride 4 -> width 8, height 8, frames, output 16
            shape = [8, 8, self.no_frames, 16]
            self.W_conv1, self.b_conv1 = self.init_weights_and_biases("W_conv1", shape)
            self.conv1 = tf.nn.conv2d(input, filter=self.W_conv1, strides=[1, 4, 4, 1], padding='VALID')

            # RELU
            self.relu1 = tf.nn.relu(self.conv1 + self.b_conv1)

            # Conv2
            # Create weights and biases
            # filters 4x4x32 stride 2 -> width 4, height 4, input 16(previous layer), output 32
            shape = [4, 4, 16, 32]
            self.W_conv2, self.b_conv2 = self.init_weights_and_biases("W_conv2", shape)
            self.conv2 = tf.nn.conv2d(self.relu1, filter=self.W_conv2, strides=[1, 2, 2, 1], padding='VALID')

            # RELU
            relu2 = tf.nn.relu(self.conv2 + self.b_conv2)

            return relu2

    ''' Reuse value prediction layer'''
    def value_trunk(self, input, reuse=False):
        with tf.device(self.device), tf.variable_scope(self.scope, reuse=reuse) as scope:
            # Reshape rnn outputs to be [1,256]
            #channels = 256 + self.lstm_additional_field
            channels = 256
            input_reshaped = tf.reshape(input, shape=[-1, channels])

            # VALUE
            shape = [channels, 1]
            self.W_value, self.b_value = self.init_weights_and_biases("W_value", shape)

            # linear
            value_nonreshape = tf.matmul(input_reshaped, self.W_value) + self.b_value
            # Reshape to be [1]
            value = tf.reshape(value_nonreshape, shape=[-1])

            return value

    ''' Reuse policy prediction layer'''

    def policy_trunk(self, input, reuse=False):
        with tf.device(self.device), tf.variable_scope(self.scope, reuse=reuse) as scope:
            # Reshape rnn outputs to be [1,256]
            #channels = 256 + self.lstm_additional_field
            channels = 256
            input_reshaped = tf.reshape(input, shape=[-1, channels])

            # POLICY
            shape = [channels, 1 * self.action_size]
            self.W_policy, self.b_policy = self.init_weights_and_biases("W_policy", shape)

            # Softmax function to give probabilities
            policy = tf.nn.softmax(tf.matmul(input_reshaped, self.W_policy) + self.b_policy)

            return policy

    ''' Create variables used for training'''
    def init_train(self):
        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            #Value loss is the sum of squared dif between targets(discounted rewards) and predictions
            #value_loss = sum((disc_rewards - value)^2)
            self.discounted_rewards = tf.placeholder(shape=[None], dtype=tf.float32,  name='discounted_rewards')
            #L2 loss
            #self.value_loss = 0.5 * tf.reduce_sum(np.square(
            # self.discounted_rewards - self.value))
            self.value_loss = 0.5 * tf.reduce_sum(tf.nn.l2_loss(
                 self.discounted_rewards - self.value))
                 
            #Entropy is used to diversify exploration
            #entropy = -sum(log(policy) * policy)
            # Avoid NaN with clipping when value in pi becomes zero
            self.entropy = -tf.reduce_sum(tf.log(self.policy + 1e-10) * \
                self.policy)

            #We need to calculate the probabilities for the selected actions in order to compute policy loss
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
            self.actions_onehot = tf.one_hot(self.actions, self.action_size, dtype=tf.float32)
            #You will have a list of the probabilities for the selected actions in self.actions
            #Take the softmax result of the policy estimate only for the selected action
            self.action_probabilities = tf.reduce_sum(self.policy * self.actions_onehot, [1])

            #Generalized Advantage Estimate = discount(rewards + gamma* values_t+1 - values_t)
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32, name='advantages')

            #Policy loss = -sum(log(policy_current_action) * advantage)
            # - because we use gradient descent not ascent
            self.policy_loss = - tf.reduce_sum(tf.log(
                self.action_probabilities + 1e-10) * self.advantages)

            # Overall loss = 0.5 value_loss + policy_loss
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy*self.beta

            self.main_loss = self.loss

            #Add auxiliary tasks loss
            self.init_auxiliary_tasks_loss(self.FLAGS)

            #GRADIENTS COMPUTATION
            #Compute variabels norm (just for plotting needs)
            if not self.FLAGS.has_vqvae_prediction:
                self.vari = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

                self.var_norm = tf.global_norm(self.vari)

                #Compute gradients of local network
                self.gradients = tf.gradients(self.loss, self.vari)
                #Compute and clip norm of gradients
                self.grad_norm = tf.global_norm(self.gradients)

                self.clipped_gradients, grad_norms = tf.clip_by_global_norm(self.gradients, 40)
                self.clipped_grad_norm = tf.global_norm(self.clipped_gradients)

                #Apply gradients to global network
                self.global_vari = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.master_network.scope)
                # self.apply_grads = self.trainer.apply_gradients(zip(self.clipped_gradients, self.global_vari))

            else:
                # decoder grads
                decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope+"/decoder")
                decoder_grads = tf.gradients(self.vqvae_loss, decoder_vars)
                decoder_grads_vars = list(zip(decoder_grads, decoder_vars))

                # embedding variables grads
                embed_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope+"/vq")
                embed_grads = tf.gradients(self.vq, embed_vars)
                embed_grads_vars = list(zip(embed_grads, embed_vars))

                # encoder grads
                encoder_vars = [x for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if x not in embed_vars and x not in decoder_vars and self.scope in x.name]
                # TODO hardcoded for now
                base_vars = [x for x in encoder_vars if 'decoder' not in x.name and 'vq' not in x.name]
                encoder_vars = [x for x in encoder_vars if 'fc' not in x.name and 'policy' not in x.name and 'value' not in x.name and 'lstm' not in x.name]
                grad_z = tf.gradients(self.recon, self.vqvae_reconstructed_cnn_output)
                encoder_grads_vars = [(tf.gradients(self.vqvae_cnn_output, var,
                                               grad_z)[0] + self.beta_vqvae *
                                  tf.gradients(self.commit, var)[0], var)
                                 for var in encoder_vars]

                # also gradients for main a3c network loss
                base_gradients = tf.gradients(self.main_loss, base_vars)
                base_gradients_grads_vars = list(zip(base_gradients, base_vars))

                # total grads
                self.grads_vars = decoder_grads_vars + embed_grads_vars + encoder_grads_vars + base_gradients_grads_vars

                self.gradients = [grad for grad,var in self.grads_vars]
                self.vars = [var for grad,var in self.grads_vars]

                # Compute and clip norm of gradients
                self.grad_norm = tf.global_norm(self.gradients)

                self.clipped_gradients, grad_norms = tf.clip_by_global_norm(self.gradients, 40)
                self.clipped_grad_norm = tf.global_norm(self.clipped_gradients)

                # Apply gradients to global network
                # get tensor by name returns ref to dtype..
                global_vari = [tf.get_default_graph().get_tensor_by_name(x.name.replace(self.scope,self.master_network.scope)) for x in self.vars]
                global_vari_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.master_network.scope)
                # Shameless hardcoding
                self.global_vari = []
                for x in global_vari:
                    for variable in global_vari_collection:
                        if variable.name == x.name:
                            self.global_vari.append(variable)

                self.apply_grads = self.trainer.apply_gradients(zip(self.clipped_gradients, self.global_vari))


                # # decoder grads
                # decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope+"/decoder")
                # decoder_grads = tf.gradients(self.recon, decoder_vars)
                # decoder_grads_vars = list(zip(decoder_grads, decoder_vars))
                #
                # # embedding variables grads
                # embed_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope+"/vq")
                # embed_grads = tf.gradients(self.recon + self.vq, embed_vars)
                # embed_grads_vars = list(zip(embed_grads, embed_vars))
                #
                # # encoder grads
                # encoder_vars = [x for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if x not in embed_vars and x not in decoder_vars and self.scope in x.name]
                # # TODO hardcoded for now
                # base_vars = [x for x in encoder_vars if 'decoder' not in x.name and 'vq' not in x.name]
                # encoder_vars = [x for x in encoder_vars if 'fc' not in x.name and 'policy' not in x.name and 'value' not in x.name and 'lstm' not in x.name]
                # # also add conv
                #
                # transferred_grads = tf.gradients(self.recon, self.vqvae_reconstructed_cnn_output)
                #
                # encoder_grads_1 = [tf.gradients(self.vqvae_cnn_output, var, transferred_grads) for var in encoder_vars]
                # encoder_grads_1 = [x[0] for x in encoder_grads_1 if x[0] is not None]
                # encoder_grads_2 = [tf.gradients(self.commit, var) for var in encoder_vars]
                # encoder_grads_2 = [x[0] for x in encoder_grads_2 if x[0] is not None]
                # encoder_grads = encoder_grads_1 + encoder_grads_2
                # encoder_grads_vars = list(zip(encoder_grads, encoder_vars))
                #
                # # also gradients for main a3c network loss
                # base_gradients = tf.gradients(self.main_loss, base_vars)
                # base_gradients_grads_vars = list(zip(base_gradients, base_vars))
                #
                # # total grads
                # self.grads_vars = decoder_grads_vars + embed_grads_vars + encoder_grads_vars + base_gradients_grads_vars
                #
                # self.gradients = [grad for grad,var in self.grads_vars]
                # self.vars = [var for grad,var in self.grads_vars]
                #
                # # Compute and clip norm of gradients
                # self.grad_norm = tf.global_norm(self.gradients)
                #
                # self.clipped_gradients, grad_norms = tf.clip_by_global_norm(self.gradients, 40)
                # self.clipped_grad_norm = tf.global_norm(self.clipped_gradients)
                #
                # # Apply gradients to global network
                # # get tensor by name returns ref to dtype..
                # global_vari = [tf.get_default_graph().get_tensor_by_name(x.name.replace(self.scope,self.master_network.scope)) for x in self.vars]
                # global_vari_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.master_network.scope)
                # # Shameless hardcoding
                # self.global_vari = []
                # for x in global_vari:
                #     for variable in global_vari_collection:
                #         if variable.name == x.name:
                #             self.global_vari.append(variable)
                #
                # self.apply_grads = self.trainer.apply_gradients(zip(self.clipped_gradients, self.global_vari))


    ''' Create desired auxiliary tasks that contribute to the overall loss '''
    def init_auxiliary_tasks(self, FLAGS):
        if FLAGS.has_reward_prediction :
            self.init_aux_task_reward_prediction()
        if FLAGS.has_pixel_control :
            self.init_aux_task_pixel_control()
        if FLAGS.has_value_prediction :
            self.init_aux_task_value_prediction()
        if FLAGS.has_frame_prediction :
            self.init_aux_task_frame_prediction()
        if FLAGS.has_action_prediction :
            self.init_aux_task_action_prediction()
        if FLAGS.has_flow_prediction :
            self.init_aux_task_flow_prediction()
        if FLAGS.has_vqvae_prediction:
            self.init_aux_task_vqvae_prediction()
        if FLAGS.has_frame_prediction_thresholded:
            self.init_aux_task_frame_prediction_thresholded()

    def init_auxiliary_tasks_loss(self, FLAGS):
        # Initialize loss and add to total loss
        if FLAGS.has_reward_prediction :
            self.init_aux_task_loss_reward_prediction()
            self.loss = self.loss + self.rp_loss
        if FLAGS.has_pixel_control :
            self.init_aux_task_loss_pixel_control()
            self.loss = self.loss + self.pc_loss
        if FLAGS.has_value_prediction :
            self.init_aux_task_loss_value_prediction()
            self.loss = self.loss + self.vp_loss
        if FLAGS.has_frame_prediction :
            self.init_aux_task_loss_frame_prediction()
            self.loss = self.loss + self.fp_loss
        if FLAGS.has_action_prediction :
            self.init_aux_task_loss_action_prediction()
            self.loss = self.loss + self.ap_loss
        if FLAGS.has_flow_prediction:
            self.init_aux_task_loss_flow_prediction()
            self.loss = self.loss + self.fl_loss
        if FLAGS.has_vqvae_prediction:
            self.init_aux_task_loss_vqvae_prediction()
            self.loss = self.loss + self.vqvae_loss
        if FLAGS.has_frame_prediction_thresholded:
            self.init_aux_task_loss_frame_prediction_thresholded()
            self.loss = self.loss + self.fp_thresh_loss

    ''' Reward prediction network
        Branches of the last conv layer into an 128 FC -> 3 FC -> Softmax'''
    def init_aux_task_reward_prediction(self):
        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            self.previous_observations_rp = tf.placeholder(shape=[None, self.input_size*self.input_size,self.no_frames],
                                                        dtype=tf.float32, name='previous_observations_rp')

            #Reshape to be a 4d tensor
            prev_obs_reshaped = tf.reshape(self.previous_observations_rp, shape=[-1, self.input_size, self.input_size, self.no_frames]
                                           , name='previous_observations_rp_reshaped')

        #We pass our frames through the CNN
        #Reuse variable
        cnn_output = self.cnn_trunk(prev_obs_reshaped, reuse=True)

        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            #FC 128 layer with relu
            shape = [9 * 9 * 32, 128]
            self.W_fc_rp1, self.b_fc_rp1 = self.init_weights_and_biases("W_fc_rp1", shape)

            # Relu2 output needs to be reshaped
            relu2_reshaped = tf.reshape(cnn_output, [-1, 9 * 9 * 32])
            self.fc_rp1 = tf.matmul(relu2_reshaped, self.W_fc_rp1) + self.b_fc_rp1
            self.relu_rp = tf.nn.relu(self.fc_rp1)

            #3-class classifier,
            shape = [128, 3]
            self.W_fc_rp2, self.b_fc_rp2 = self.init_weights_and_biases("W_fc_rp2", shape)
            self.fc_rp2 = tf.matmul(self.relu_rp, self.W_fc_rp2) + self.b_fc_rp2

            # Zero reward, negative reward, positive reward = 3 values for one-hot v np.clipector
            self.rp_prediction = tf.nn.softmax(self.fc_rp2)

    def init_aux_task_pixel_control(self):
        print ('Auxiliary task pixel control only available for LSTM network')
        raise NotImplementedError

    def init_aux_task_value_prediction(self):
        print ('Auxiliary task value prediction only available for LSTM network')
        raise NotImplementedError

    def init_aux_task_loss_reward_prediction(self):
        self.reward_prediction_target = tf.placeholder(dtype=tf.float32,shape=[None,3], name='reward_prediction_target')

        #Clip the predicted reward
        #TODO WHY?
        rp_prediction_clip = tf.clip_by_value(self.rp_prediction, 0.0, 1.0)

        #If the sigmoid gives 1 probability then log(1) = 0 so you have 0 loss
        #Multiplying by target means that you care only that you did not predict the correct one
        #You ignore the loss for predicting as the incorect values
        #Because it is one hot encoded, and not regression. We do not care about the reward value
        rp_loss = -tf.reduce_sum(self.reward_prediction_target * tf.log(
            rp_prediction_clip + 1e-10))

        #empirically, some values of loss are much larger than others
        #leads to destabilizing

        self.rp_loss = rp_loss * self.FLAGS.rp_loss_lambda

    def init_aux_task_loss_pixel_control(self):
        print('Auxiliary task pixel control only available for LSTM network')
        raise NotImplementedError

    def init_aux_task_loss_value_prediction(self):
        print ('Auxiliary task value prediction only available for LSTM network')
        raise NotImplementedError

    def init_aux_task_frame_prediction(self):
        print ('Auxiliary task frame prediction only available for LSTM network')
        raise NotImplementedError

''' A3C Network that uses LSTM '''
class A3CLSTM(BaseNetwork):

    def __init__(self, scope, trainer,master_network, FLAGS):
        '''
            @scope : scope in which to create variables
            @trainer : trainer used to minimize loss
            @master_network : master network
            @FLAGS : specify different options through flags
        '''
        BaseNetwork.__init__(self,scope,trainer,master_network,FLAGS)

        #Construct network in current scope
        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            lstm_input = self.fc_output

            #Action concat to LSTM
            if self.FLAGS.concat_action_lstm:
                self.previous_observations_actions =  tf.placeholder(shape=[
                    None], dtype=tf.int32, \
                    name='action_placeholder_for_lstm_in_base')

                prev_act_reshaped = tf.one_hot(self.previous_observations_actions,
                                       self.action_size,
                                       name='one_hot_action_for_lstm')

                lstm_input = tf.concat([self.fc_output, prev_act_reshaped],
                                       axis=1)

            #The network up to this point is initialized by the BaseNetwork

            sequence_length = tf.shape(self.input_reshaped)[:1]

            # We need to initialize the lstm state in order to use it later
            #channels = 256 + self.lstm_additional_field
            channels = 256
            c_state = np.zeros([1, channels])
            h_state = np.zeros([1, channels])
            self.lstm_state_init = tf.contrib.rnn.LSTMStateTuple(c_state,h_state)

            # LSTM
            self.lstm = tf.contrib.rnn.BasicLSTMCell(channels, state_is_tuple=True)
            # Create initial state for lstm
            self.initial_cell_c_state = tf.placeholder(dtype=tf.float32,
                                                       shape=[1, channels],
                                                       name='init_cell_c_state')
            self.initial_cell_h_state = tf.placeholder(dtype=tf.float32,
                                                       shape=[1, channels],
                                                       name='init_cell_h_state')
            self.lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_cell_c_state, self.initial_cell_h_state)

        rnn_outputs, self.lstm_state_out = self.lstm_trunk(lstm_input, sequence_length, self.lstm_state)

        #POLICY
        self.policy = self.policy_trunk(rnn_outputs)

        #VALUE
        self.value = self.value_trunk(rnn_outputs)

        #Initialize auxiliary tasks
        self.init_auxiliary_tasks(FLAGS)

        #TODO ELU INSTEAD OF RELU
        if self.scope != 'global':
            #Initialize losses
            self.init_train()


    ''' Reuse lstm + cnn layers for auxiliary tasks'''
    def lstm_trunk(self, input, sequence_length, lstm_state, reuse=False):
        with tf.device(self.device), tf.variable_scope(self.scope, reuse=reuse) as scope:
            # Reshape output to be ?, lstm channels
            # 3d tensor to use with dynamic rnn which needs [batch_size, max_time, ...]

            channels = 256 + self.lstm_additional_field
            input_reshaped = tf.reshape(input, shape=[1, -1, channels])

            # tf.nn.dynamic_rnn
            # time_major == False (default), this must be a Tensor of shape: [batch_size, max_time, ...]
            rnn_outputs, rnn_state = tf.nn.dynamic_rnn(self.lstm, input_reshaped,
                                                                 sequence_length=sequence_length,
                                                                 initial_state=lstm_state,
                                                                 time_major=False, scope=scope)

            lstm_state_out = (rnn_state[0][:1, :], rnn_state[1][:1, :])

            return rnn_outputs, lstm_state_out

    ''' Action prediction network reuses the LSTM (with sampled experience)'''
    def init_aux_task_action_prediction(self):
        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            self.first_state_ap = tf.placeholder(shape=[None, self.input_size *
                                           self.input_size ,self.no_frames],
                                                           dtype=tf.float32,
                                              name='first_state_ap')

            # Reshape to be a 4d tensor
            first_state_reshaped = tf.reshape(self.first_state_ap,
                                           shape=[-1, self.input_size, self.input_size, self.no_frames]
                                           , name='first_state_ap_reshaped')

            self.second_state_ap = tf.placeholder(shape=[None, self.input_size *
                                                     self.input_size,
                                                     self.no_frames],
                                              dtype=tf.float32,
                                              name='second_state_ap')

            # Reshape to be a 4d tensor
            second_state_reshaped = tf.reshape(self.second_state_ap,
                                              shape=[-1, self.input_size,
                                                     self.input_size,
                                                     self.no_frames]
                                              , name='second_state_ap_reshaped')

        # We pass our frames through the CNN
        # Reuse variable
        cnn_output_first = self.cnn_trunk(first_state_reshaped, reuse=True)
        cnn_output_second = self.cnn_trunk(second_state_reshaped, reuse=True)

        #We pass it through the FC layer
        fc_output_first = self.fc_trunk(cnn_output_first, reuse=True)
        fc_output_second = self.fc_trunk(cnn_output_second, reuse=True)

        #As opposed to the reward prediction, we also reuse the LSTM
        sequence_length = tf.shape(first_state_reshaped)[:1]

        #Initial state of the lstm should be 0?
        #Reset lstm state
        zero_state = self.lstm.zero_state(1, tf.float32)

        rnn_outputs_first, rnn_state = self.lstm_trunk(fc_output_first,
                                                   sequence_length, zero_state, reuse=True)

        rnn_outputs_second, rnn_state = self.lstm_trunk(fc_output_second,
                                                        sequence_length,
                                                        zero_state, reuse=True)

        #Predict action
        #Concat channels
        rnn_outputs = tf.concat([rnn_outputs_first, rnn_outputs_second],axis=2)
        #channels = 256 + self.lstm_additional_field
        channels = 256
        rnn_outputs = tf.reshape(rnn_outputs, [-1,1,1, channels*2])

        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            #Conv 256
            shape = [1, 1, channels*2, channels]
            W_conv1, b_conv1 = self.init_weights_and_biases("W_conv1_ap",
                                                                      shape)
            conv1 = tf.nn.conv2d(rnn_outputs, filter=W_conv1,
                                      strides=[1, 2, 2, 1], padding='VALID')

            # RELU
            relu1 = tf.nn.relu(conv1 + b_conv1)
            relu1 = tf.reshape(relu1,[-1, channels])

            with tf.variable_scope('Aux_AP') as scope:
                # LSTM
                # Create initial state for lstm
                self.initial_cell_c_state_ap = tf.placeholder(dtype=tf.float32,
                                                           shape=[1, channels],
                                                           name='init_cell_c_state_ap')
                self.initial_cell_h_state_ap = tf.placeholder(dtype=tf.float32,
                                                           shape=[1, channels],
                                                           name='init_cell_h_state_ap')
                self.lstm_ap = tf.contrib.rnn.BasicLSTMCell(channels,
                                                            state_is_tuple=True)

                self.lstm_state_ap = tf.contrib.rnn.LSTMStateTuple(
                    self.initial_cell_c_state_ap, self.initial_cell_h_state_ap)


                sequence_length = tf.shape(relu1)[:1]

                relu1 = tf.reshape(relu1, shape=[1, -1, channels])


                # Initial state of the lstm should be 0?
                # Reset lstm state
                zero_state = self.lstm.zero_state(1, tf.float32)

                # tf.nn.dynamic_rnn
                # time_major == False (default), this must be a Tensor of shape: [batch_size, max_time, ...]
                rnn_outputs, rnn_state = tf.nn.dynamic_rnn(self.lstm_ap,
                                                           relu1,
                                                                     sequence_length=sequence_length,
                                                                     initial_state=zero_state,
                                                                     time_major=False, scope=scope)

                self.lstm_state_out_ap = (rnn_state[0][:1, :], rnn_state[1][:1, :])




            #predict action
            shape = [channels, 1 * self.action_size]
            W_action, b_action = self.init_weights_and_biases("W_fc_ap",
                                                                        shape)

            rnn_outputs = tf.reshape(rnn_outputs,[-1,channels])
            # Softmax function to give probabilities in loss
            self.ap_prediction = tf.matmul(rnn_outputs, W_action) + b_action



        #TODO ADD LSTM HERE INSTEAD OF CONV?
    ''' Value prediction network reuses the LSTM and value prediction
        (with sampled experience)'''
    def init_aux_task_value_prediction(self):
        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            self.previous_observations_vp = tf.placeholder(shape=[None, self.input_size * self.input_size ,self.no_frames],
                                                           dtype=tf.float32, name='previous_observations_vp')

            # Reshape to be a 4d tensor
            prev_obs_reshaped = tf.reshape(self.previous_observations_vp,
                                           shape=[-1, self.input_size, self.input_size, self.no_frames]
                                           , name='previous_observations_vp_reshaped')

            if self.FLAGS.concat_action_lstm:
                self.previous_observations_vp_actions =  tf.placeholder(shape=[
                    None], dtype=tf.int32, \
                    name='action_placeholder_for_lstm')

                prev_act_reshaped = tf.one_hot(self.previous_observations_vp_actions,
                                       self.action_size,
                                       name='one_hot_action_for_vp')

        # We pass our frames through the CNN
        # Reuse variable
        cnn_output = self.cnn_trunk(prev_obs_reshaped, reuse=True)

        #We pass it through the FC layer
        fc_output = self.fc_trunk(cnn_output, reuse=True)
        lstm_input = fc_output

        if self.FLAGS.concat_action_lstm:
            #Concatenate actions so as to pass to LSTM
            lstm_input = tf.concat([fc_output, prev_act_reshaped], 1)

        #As opposed to the reward prediction, we also reuse the LSTM
        sequence_length = tf.shape(prev_obs_reshaped)[:1]

        #Initial state of the lstm should be 0?
        #Reset lstm state
        zero_state = self.lstm.zero_state(1, tf.float32)

        rnn_outputs, rnn_state = self.lstm_trunk(lstm_input, sequence_length, zero_state, reuse=True)

        #Predict value
        self.vp_prediction = self.value_trunk(rnn_outputs, reuse=True)


    ''' Frame prediction network reuses the LSTM (with sampled experience)'''
    def init_aux_task_frame_prediction(self):
        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            self.previous_observations_fp = tf.placeholder(
                shape=[None, self.input_size * self.input_size,
                       self.no_frames],
                dtype=tf.float32, name='previous_observations_fp')

            # Reshape to be a 4d tensor
            prev_obs_reshaped = tf.reshape(self.previous_observations_fp,
                                           shape=[-1, self.input_size, self.input_size, self.no_frames]
                                           ,
                                           name='previous_observations_fp_reshaped')
        if self.FLAGS.concat_action_lstm:
            self.previous_observations_fp_actions = tf.placeholder(shape=[
                None], dtype=tf.int32, \
                name='fp_action_placeholder_for_lstm')

            prev_act_reshaped = tf.one_hot(
                self.previous_observations_fp_actions,
                self.action_size,
                name='one_hot_action_for_fp')



        # We pass our frames through the CNN
        # Reuse variable
        cnn_output = self.cnn_trunk(prev_obs_reshaped, reuse=True)

        # # We pass it through the FC layer
        fc_output = self.fc_trunk(cnn_output, reuse=True)
        lstm_input = fc_output
        # fc_output = self.fc_trunk(cnn_output, reuse=True)
        if self.FLAGS.concat_action_lstm:
        # lstm_input = fc_output
            lstm_input = tf.concat([fc_output, prev_act_reshaped], 1)
        #
        # if self.FLAGS.concat_action_lstm:
        sequence_length = tf.shape(prev_obs_reshaped)[:1]
        #     # Concatenate actions so as to pass to LSTM
        #     lstm_input = tf.concat([fc_output, prev_act_reshaped], 1)
        #
        zero_state = self.lstm.zero_state(1, tf.float32)
        # # As opposed to the reward prediction, we also reuse the LSTM
        rnn_outputs, rnn_state = self.lstm_trunk(lstm_input, sequence_length, zero_state, reuse=True)
        # sequence_length = tf.shape(prev_obs_reshaped)[:1]
        #
        channels = 256
        rnn_outputs = tf.reshape(rnn_outputs, [-1, channels])

        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            # # Map lstm output to 9x9x32 to reconstruct the cnn trunk output
            shape = [channels, 9 * 9 * 32]
            W_fc_fp1, b_fc_fp1 = self.init_weights_and_biases("W_fc_fp1", shape)
            fc_fp = tf.nn.relu(tf.matmul(rnn_outputs, W_fc_fp1) + b_fc_fp1)
            fc_fp = tf.reshape(fc_fp, [-1, 9, 9, 32])

            # Deconv layer map to 20x20x16
            shape = [4, 4, 16, 32]
            W_deconv_fp1, b_deconv_fp1 = self.init_weights_and_biases(
                "W_deconv_fp1", shape)

            # TODO
            padding_type = 'VALID'
            if padding_type == 'VALID':
                out_height = (fc_fp.shape[1].value - 1) * 2 + \
                             W_deconv_fp1.get_shape()[0].value
                out_width = (fc_fp.shape[2].value - 1) * 2 + \
                            W_deconv_fp1.get_shape()[1].value
                out_shape = [tf.shape(fc_fp)[0], out_height, out_width,
                             W_deconv_fp1.get_shape()[
                                 2].value]
            fp_deconv1 = tf.nn.conv2d_transpose(fc_fp,
                                                     filter=W_deconv_fp1,
                                                     output_shape=out_shape,
                                                     strides=[1, 2, 2, 1],
                                                     padding='VALID')

            # 20x20x16
            fp_deconv1 = tf.nn.relu(fp_deconv1 + b_deconv_fp1)

            # Deconv layer map to 84x84x1
            shape = [8, 8, self.no_frames, 16]
            W_deconv_fp2, b_deconv_fp2 = self.init_weights_and_biases(
                "W_deconv_fp2", shape)

            # TODO
            padding_type = 'VALID'
            if padding_type == 'VALID':
                out_height = (out_height - 1) * 4 + \
                             W_deconv_fp2.get_shape()[0].value
                out_width = (out_width - 1) * 4 + \
                            W_deconv_fp2.get_shape()[1].value
                out_shape = [tf.shape(fp_deconv1)[0], out_height, out_width,
                             W_deconv_fp2.get_shape()[
                                 2].value]
            fp_deconv2 = tf.nn.conv2d_transpose(fp_deconv1,
                                                filter=W_deconv_fp2,
                                                output_shape=out_shape,
                                                strides=[1, 4, 4, 1],
                                                padding='VALID')

            # 84,84,1
            self.fp_reconstruction = fp_deconv2 + b_deconv_fp2

            # self.fp_reconstruction = tf.nn.relu(fp_deconv2 + b_deconv_fp2)

    def init_aux_task_frame_prediction_thresholded(self):
        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            self.previous_observations_fp_thresh = tf.placeholder(
                shape=[None, self.input_size * self.input_size,
                       self.no_frames],
                dtype=tf.float32, name='previous_observations_fp_thresh')

            self.current_observations_fp_thresh = tf.placeholder(
                shape=[None, self.input_size * self.input_size,
                       self.no_frames],
                dtype=tf.float32, name='current_observations_fp_thresh')

            # Reshape to be a 4d tensor
            prev_obs_reshaped = tf.reshape(self.previous_observations_fp_thresh,
                                           shape=[-1, self.input_size, self.input_size, self.no_frames]
                                           ,
                                           name='previous_observations_fp_thresh_reshaped')

        # We pass our frames through the CNN
        # Reuse variable
        cnn_output = self.cnn_trunk(prev_obs_reshaped, reuse=True)

        # # We pass it through the FC layer
        fc_output = self.fc_trunk(cnn_output, reuse=True)
        lstm_input = fc_output
        # fc_output = self.fc_trunk(cnn_output, reuse=True)


        sequence_length = tf.shape(prev_obs_reshaped)[:1]

        # Initial state of the lstm should be 0?
        # Reset lstm state
        zero_state = self.lstm.zero_state(1, tf.float32)

        rnn_outputs, rnn_state = self.lstm_trunk(lstm_input, sequence_length, zero_state, reuse=True)

        #channels = 256 + self.lstm_additional_field
        channels = 256
        rnn_outputs = tf.reshape(rnn_outputs, [-1, channels])

        with tf.device(self.device), tf.variable_scope(self.scope) as scope:

            shape = [channels, 9 * 9 * 32]
            W_fc_fp1, b_fc_fp1 = self.init_weights_and_biases(
                "W_fc_fp_thresh1", shape)
            fc_fp = tf.nn.relu(tf.matmul(rnn_outputs, W_fc_fp1) + b_fc_fp1)
            fc_fp = tf.reshape(fc_fp, [-1, 9, 9, 32])

            # Deconv layer map to 20x20x16
            shape = [4, 4, 16, 32]
            W_deconv_fp1, b_deconv_fp1 = self.init_weights_and_biases(
                "W_deconv_fp_thresh1", shape)

            # TODO
            padding_type = 'VALID'
            if padding_type == 'VALID':
                out_height = (fc_fp.shape[1].value - 1) * 2 + \
                             W_deconv_fp1.get_shape()[0].value
                out_width = (fc_fp.shape[2].value - 1) * 2 + \
                            W_deconv_fp1.get_shape()[1].value
                out_shape = [tf.shape(fc_fp)[0], out_height, out_width,
                             W_deconv_fp1.get_shape()[
                                 2].value]
            fp_deconv1 = tf.nn.conv2d_transpose(fc_fp,
                                                     filter=W_deconv_fp1,
                                                     output_shape=out_shape,
                                                     strides=[1, 2, 2, 1],
                                                     padding='VALID')

            # 20x20x16
            fp_deconv1 = tf.nn.relu(fp_deconv1 + b_deconv_fp1)

            # Deconv layer map to 84x84x1
            shape = [8, 8, self.no_frames, 16]
            W_deconv_fp2, b_deconv_fp2 = self.init_weights_and_biases(
                "W_deconv_fp_thresh2", shape)

            # TODO
            padding_type = 'VALID'
            if padding_type == 'VALID':
                out_height = (out_height - 1) * 4 + \
                             W_deconv_fp2.get_shape()[0].value
                out_width = (out_width - 1) * 4 + \
                            W_deconv_fp2.get_shape()[1].value
                out_shape = [tf.shape(fp_deconv1)[0], out_height, out_width,
                             W_deconv_fp2.get_shape()[
                                 2].value]
            fp_deconv2 = tf.nn.conv2d_transpose(fp_deconv1,
                                                filter=W_deconv_fp2,
                                                output_shape=out_shape,
                                                strides=[1, 4, 4, 1],
                                                padding='VALID')

            # 84,84,1
            self.fp_thresh_reconstruction = tf.nn.relu(fp_deconv2 + b_deconv_fp2)

            # visualization
            self.fp_thresh_reconstruction_vis = self.fp_thresh_reconstruction


    ''' Frame prediction network reuses the LSTM (with sampled experience)'''

    def init_aux_task_flow_prediction(self):
        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            self.previous_observations_fl = tf.placeholder(
                shape=[None, self.input_size * self.input_size,
                       self.no_frames],
                dtype=tf.float32, name='previous_observations_fl')

            # Reshape to be a 4d tensor
            prev_obs_reshaped = tf.reshape(self.previous_observations_fl,
                                           shape=[-1, self.input_size,
                                                  self.input_size,
                                                  self.no_frames]
                                           ,
                                           name='previous_observations_fl_reshaped')

            self.previous_observations_fl_next = tf.placeholder(
                shape=[None, self.input_size * self.input_size,
                       self.no_frames],
                dtype=tf.float32, name='previous_observations_fl_next')

        # We pass our frames through the CNN
        # Reuse variable
        cnn_output = self.cnn_trunk(prev_obs_reshaped, reuse=True)

        # We pass it through the FC layer
        fc_output = self.fc_trunk(cnn_output, reuse=True)
        lstm_input = fc_output

        # As opposed to the reward prediction, we also reuse the LSTM
        sequence_length = tf.shape(prev_obs_reshaped)[:1]

        # Initial state of the lstm should be 0?
        # Reset lstm state
        zero_state = self.lstm.zero_state(1, tf.float32)

        rnn_outputs, rnn_state = self.lstm_trunk(lstm_input, sequence_length,
                                                 zero_state, reuse=True)

        # channels = 256 + self.lstm_additional_field
        channels = 256
        rnn_outputs = tf.reshape(rnn_outputs, [-1, channels])

        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            # Map lstm output to 9x9x32 to reconstruct the cnn trunk output
            shape = [channels, 9 * 9 * 32]
            W_fc_fp1, b_fc_fp1 = self.init_weights_and_biases("W_fc_fl1", shape)
            fc_fp = tf.nn.relu(tf.matmul(rnn_outputs, W_fc_fp1) + b_fc_fp1)
            fc_fp = tf.reshape(fc_fp, [-1, 9, 9, 32])

            # Deconv layer map to 20x20x16
            shape = [4, 4, 16, 32]
            W_deconv_fp1, b_deconv_fp1 = self.init_weights_and_biases(
                "W_deconv_fl1", shape)

            # TODO
            padding_type = 'VALID'
            if padding_type == 'VALID':
                out_height = (fc_fp.shape[1].value - 1) * 2 + \
                             W_deconv_fp1.get_shape()[0].value
                out_width = (fc_fp.shape[2].value - 1) * 2 + \
                            W_deconv_fp1.get_shape()[1].value
                out_shape = [tf.shape(fc_fp)[0], out_height, out_width,
                             W_deconv_fp1.get_shape()[
                                 2].value]
            fp_deconv1 = tf.nn.conv2d_transpose(fc_fp,
                                                filter=W_deconv_fp1,
                                                output_shape=out_shape,
                                                strides=[1, 2, 2, 1],
                                                padding='VALID')

            # 20x20x16
            fp_deconv1 = tf.nn.relu(fp_deconv1 + b_deconv_fp1)

            # Deconv layer map to 84x84x2
            # 2 is for flow
            shape = [8, 8, 2, 16]
            W_deconv_fp2, b_deconv_fp2 = self.init_weights_and_biases(
                "W_deconv_fl2", shape)

            # TODO
            padding_type = 'VALID'
            if padding_type == 'VALID':
                out_height = (out_height - 1) * 4 + \
                             W_deconv_fp2.get_shape()[0].value
                out_width = (out_width - 1) * 4 + \
                            W_deconv_fp2.get_shape()[1].value
                out_shape = [tf.shape(fp_deconv1)[0], out_height, out_width,
                             W_deconv_fp2.get_shape()[
                                 2].value]
            fp_deconv2 = tf.nn.conv2d_transpose(fp_deconv1,
                                                filter=W_deconv_fp2,
                                                output_shape=out_shape,
                                                strides=[1, 4, 4, 1],
                                                padding='VALID')

            # 84,84,2
            self.flow_prediction = tf.nn.relu(fp_deconv2 + b_deconv_fp2)

    ''' Frame prediction network reuses the LSTM (with sampled experience)'''

    def init_aux_task_vqvae_prediction(self):
        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            self.previous_observations_vqvae = tf.placeholder(
                shape=[None, self.input_size * self.input_size,
                       self.no_frames],
                dtype=tf.float32, name='previous_observations_vqvae')

            # Reshape to be a 4d tensor
            prev_obs_reshaped = tf.reshape(self.previous_observations_vqvae,
                                           shape=[-1, self.input_size,
                                                  self.input_size,
                                                  self.no_frames]
                                           ,
                                           name='previous_observations_vqvae_reshaped')

        # We pass our frames through the CNN
        # Reuse variable
        self.vqvae_cnn_output = self.cnn_trunk(prev_obs_reshaped, reuse=True)
        cnn_output = self.vqvae_cnn_output
        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            # VQ vector quantization
            # slit cnn output into embeddings and embedding indices
            with tf.variable_scope("vq"):
                K = 512
                D = 32
                embeds = tf.get_variable('embed', [K, D],
                                         initializer=tf.truncated_normal_initializer(stddev=0.01))

                # reconstruct cnn output based on embeddings and embedding indices
                self.embeds = embeds

                # Nearest neighbour
                vq_in = tf.expand_dims(cnn_output, axis=-2)
                distances = tf.norm(vq_in - self.embeds, axis=-1)
                k = tf.argmin(distances, axis=-1)  # -> [latent_h,latent_w]
                self.vqvae_indices = k

                if self.scope != 'global':
                    # Visualize histogram of chosen indices (of embeddings)
                    reshaped_indices = tf.reshape(self.vqvae_indices, [-1])
                    self.vqvae_indices_ids,_, self.vqvae_indices_ids_count = tf.unique_with_counts(reshaped_indices)


                reconstructed_cnn_output = tf.gather(self.embeds, k)
                self.vqvae_reconstructed_cnn_output = reconstructed_cnn_output

            with tf.variable_scope("decoder"):
                # Deconv layer map to 20x20x16
                shape = [4, 4, 16, 32]
                W_deconv_fp1, b_deconv_fp1 = self.init_weights_and_biases(
                    "W_deconv_fl1", shape)

                # TODO
                padding_type = 'VALID'
                if padding_type == 'VALID':
                    out_height = (reconstructed_cnn_output.shape[1].value - 1) * 2 + \
                                 W_deconv_fp1.get_shape()[0].value
                    out_width = (reconstructed_cnn_output.shape[2].value - 1) * 2 + \
                                W_deconv_fp1.get_shape()[1].value
                    out_shape = [tf.shape(reconstructed_cnn_output)[0], out_height, out_width,
                                 W_deconv_fp1.get_shape()[
                                     2].value]
                fp_deconv1 = tf.nn.conv2d_transpose(reconstructed_cnn_output,
                                                    filter=W_deconv_fp1,
                                                    output_shape=out_shape,
                                                    strides=[1, 2, 2, 1],
                                                    padding='VALID')

                # 20x20x16
                fp_deconv1 = tf.nn.relu(fp_deconv1 + b_deconv_fp1)

                # Deconv layer map to 84x84x2
                # 1 is for grayscale image
                shape = [8, 8, 1, 16]
                W_deconv_fp2, b_deconv_fp2 = self.init_weights_and_biases(
                    "W_deconv_fl2", shape)

                # TODO
                padding_type = 'VALID'
                if padding_type == 'VALID':
                    out_height = (out_height - 1) * 4 + \
                                 W_deconv_fp2.get_shape()[0].value
                    out_width = (out_width - 1) * 4 + \
                                W_deconv_fp2.get_shape()[1].value
                    out_shape = [tf.shape(fp_deconv1)[0], out_height, out_width,
                                 W_deconv_fp2.get_shape()[
                                     2].value]
                fp_deconv2 = tf.nn.conv2d_transpose(fp_deconv1,
                                                    filter=W_deconv_fp2,
                                                    output_shape=out_shape,
                                                    strides=[1, 4, 4, 1],
                                                    padding='VALID')

                # 84,84,1
                self.vqvae_prediction = tf.nn.relu(fp_deconv2 + b_deconv_fp2)


    ''' Pixel control network reuses the LSTM and value prediction
            (with sampled experience)'''
    def init_aux_task_pixel_control(self):
        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            self.previous_observations_pc = tf.placeholder(
                shape=[None, self.input_size * self.input_size, self.no_frames],
                dtype=tf.float32, name='previous_observations_pc')

            # Reshape to be a 4d tensor
            prev_obs_reshaped = tf.reshape(self.previous_observations_pc,
                                           shape=[-1, self.input_size, self.input_size, self.no_frames]
                                           , name='previous_observations_pc_reshaped')

        # We pass our frames through the CNN
        # Reuse variable
        cnn_output = self.cnn_trunk(prev_obs_reshaped, reuse=True)

        # We pass it through the FC layer
        fc_output = self.fc_trunk(cnn_output, reuse=True)

        # As opposed to the reward prediction, we also reuse the LSTM
        sequence_length = tf.shape(prev_obs_reshaped)[:1]

        # Initial state of the lstm should be 0?
        # Reset lstm state
        zero_state = self.lstm.zero_state(1, tf.float32)

        rnn_outputs, rnn_state = self.lstm_trunk(fc_output, sequence_length, zero_state, reuse=True)

        rnn_outputs = tf.reshape(rnn_outputs, [-1, 256])

        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            # Map lstm output to 9x9x32 to reconstruct cnn trunk output
            shape = [256, 9 * 9 * 32]
            W_fc_pc1, b_fc_pc1 = self.init_weights_and_biases("W_fc_pc1", shape)
            fc_pc = tf.nn.relu(tf.matmul(rnn_outputs, W_fc_pc1) + b_fc_pc1)
            fc_pc = tf.reshape(fc_pc, [-1, 9, 9, 32])


            # Deconv layer for value
            shape = [4, 4, 1, 32]
            W_deconv_value, b_deconv_value = self.init_weights_and_biases("W_deconv_value", shape)

            #TODO
            padding_type ='VALID'
            if padding_type == 'VALID':
                out_height = (fc_pc.shape[1].value - 1) * 2 + \
                             W_deconv_value.get_shape()[0].value
                out_width = (fc_pc.shape[2].value - 1) * 2 + W_deconv_value.get_shape()[1].value
                out_shape = [tf.shape(fc_pc)[0], out_height, out_width,
                             W_deconv_value.get_shape()[
                    2].value]
            pc_value_deconv = tf.nn.conv2d_transpose(fc_pc,filter=W_deconv_value,
                                                     output_shape=out_shape,strides=[1,2,2,1],padding='VALID')

            #20x20x1
            pc_value_deconv = tf.nn.relu(pc_value_deconv + b_deconv_value)

            # Deconv layer for advantage
            shape = [4, 4, self.action_size, 32]
            W_deconv_advantage, b_deconv_advantage = self.init_weights_and_biases("W_deconv_advantage", shape)
            # TODO
            padding_type = 'VALID'
            if padding_type == 'VALID':
                out_height = (fc_pc.shape[1].value - 1) * 2 + W_deconv_advantage.get_shape()[0].value
                out_width = (fc_pc.shape[2].value - 1) * 2 + W_deconv_advantage.get_shape()[1].value
                out_shape = [tf.shape(fc_pc)[0], out_height, out_width,
                             W_deconv_advantage.get_shape()[2].value]
            pc_advantage_deconv = tf.nn.conv2d_transpose(fc_pc, filter=W_deconv_advantage,
                                                     output_shape=out_shape, strides=[1, 2, 2, 1],
                                                     padding='VALID')
            #20x20x3
            pc_advantage_deconv = tf.nn.relu(pc_advantage_deconv + b_deconv_advantage)

            #Dueling parametrization of Wang
            # Advantage mean
            pc_advantage_deconv_mean = tf.reduce_mean(pc_advantage_deconv, reduction_indices=3, keep_dims=True)

            # Pixel change Q values
            self.pc_q = pc_value_deconv + pc_advantage_deconv - pc_advantage_deconv_mean

            # Max Q value
            #20x20
            self.pc_q_max = tf.reduce_max(self.pc_q, reduction_indices=3, keep_dims=False)


    ''' Value prediction loss '''
    def init_aux_task_loss_value_prediction(self):
        self.value_prediction_target = tf.placeholder(dtype=tf.float32,shape=[None],name='value_prediction_target')

        #same as with value loss in main network
        #vp_loss = tf.reduce_sum(np.square(self.value_prediction_target -
        #                                       self.vp_prediction))

        vp_loss = tf.reduce_sum(tf.nn.l2_loss(self.value_prediction_target -
                                              self.vp_prediction))

        self.vp_loss = vp_loss * self.FLAGS.vp_loss_lambda

    ''' Action prediction loss '''
    def init_aux_task_loss_action_prediction(self):
        self.action_prediction_target = tf.placeholder(dtype=tf.int32,
                                                      shape=[None],
                                                      name='action_prediction_target')
        # convert actions to onehot
        action_target = tf.one_hot(self.action_prediction_target,
                                   self.action_size, name='one_hot_action_ap')

        ap_loss = tf.losses.softmax_cross_entropy(
            action_target, self.ap_prediction,scope="ap_loss")

        self.ap_loss = ap_loss * self.FLAGS.ap_loss_lambda

    ''' Pixel control loss '''

    def init_aux_task_loss_frame_prediction(self):
        self.fp_previous_frames_target = tf.placeholder(dtype=tf.float32,
                                               shape=[None, self.input_size *
                                                      self.input_size,
                                                      self.no_frames])
        fp_previous_frames_target_reshaped = tf.reshape(self.fp_previous_frames_target,
                                       shape=[-1, self.input_size,
                                              self.input_size, self.no_frames]
                                       ,
                                       name='fp_previous_frames_target_reshaped')

        # Check whether we predict the next frame or the difference between
        # frames
        if self.FLAGS.has_frame_dif_prediction:
            real_difference = self.fp_previous_frames_target - self.previous_observations_fp
            reshaped_difference = tf.reshape(real_difference, [-1,
                                                               self.input_size,
                                                               self.input_size,
                                                               self.no_frames])
            fp_loss = tf.reduce_sum(tf.nn.l2_loss(reshaped_difference -
                                                  self.fp_reconstruction))
        else:
            # Log because the loss is very high
            # fp_loss = tf.reduce_sum(np.square(self.q_target - q_value))
            fp_loss = tf.reduce_sum(tf.nn.l2_loss(fp_previous_frames_target_reshaped -
                                                  self.fp_reconstruction))

        self.fp_loss = fp_loss * self.FLAGS.fp_loss_lambda

    def init_aux_task_loss_frame_prediction_thresholded(self):
        self.fp_thresh_previous_frames_target = tf.placeholder(dtype=tf.float32,
                                               shape=[None, self.input_size *
                                                      self.input_size,
                                                      self.no_frames])
        fp_thresh_previous_frames_target_reshaped = tf.reshape(
            self.fp_thresh_previous_frames_target,
                                       shape=[-1, self.input_size,
                                              self.input_size, self.no_frames]
                                       ,
                                       name='fp_thresh_previous_frames_target_reshaped')


        # Log because the loss is very high
        # fp_loss = tf.reduce_sum(np.square(self.q_target - q_value))
        fp_thresh_loss = tf.reduce_sum(tf.nn.l2_loss(
            fp_thresh_previous_frames_target_reshaped -
                                              self.fp_thresh_reconstruction))

        self.fp_thresh_loss = fp_thresh_loss * self.FLAGS.fp_thresh_loss_lambda

    def init_aux_task_loss_flow_prediction(self):
        self.fl_target = tf.placeholder(dtype=tf.float32,
                                               shape=[None, self.input_size *
                                                      self.input_size,
                                                      2])
        fl_target_reshaped = tf.reshape(self.fl_target,
                                       shape=[-1, self.input_size,
                                              self.input_size, 2]
                                       ,name='fl_target_reshaped')


        # Gives 0 values
        flow_loss = tf.reduce_sum(tf.nn.l2_loss(fl_target_reshaped -
                                                 self.flow_prediction))
        self.fl_loss = flow_loss * self.FLAGS.fl_loss_lambda


        # flow_loss = tf.reduce_sum(tf.abs(fl_target_reshaped - self.flow_prediction))
        # self.fl_loss = flow_loss * self.FLAGS.fl_loss_lambda

    def init_aux_task_loss_vqvae_prediction(self):
        self.vqvae_target = tf.placeholder(dtype=tf.float32,
                                               shape=[None, self.input_size *
                                                      self.input_size,
                                                      self.no_frames])
        vqvae_target_reshaped = tf.reshape(self.vqvae_target,
                                       shape=[-1, self.input_size,
                                              self.input_size, self.no_frames]
                                       ,
                                       name='vqvae_target_target_reshaped')

        # vqvae_reconstruction_loss = tf.reduce_sum(tf.nn.l2_loss(vqvae_target_reshaped -
        #                                       self.vqvae_prediction))


        self.recon = tf.reduce_mean((self.vqvae_prediction - vqvae_target_reshaped) ** 2, axis=[0, 1, 2, 3])
        self.vq = tf.reduce_mean(
            tf.norm(tf.stop_gradient(self.vqvae_cnn_output) - self.vqvae_reconstructed_cnn_output, axis=-1) ** 2,
            axis=[0, 1, 2])
        self.commit = tf.reduce_mean(
            tf.norm(self.vqvae_cnn_output - tf.stop_gradient(self.vqvae_reconstructed_cnn_output), axis=-1) ** 2,
            axis=[0, 1, 2])

        self.beta_vqvae = 0.25

        self.vqvae_loss = self.recon + self.vq + self.beta_vqvae * self.commit

        self.vqvae_loss = self.vqvae_loss * self.FLAGS.vqvae_loss_lambda



    ''' Pixel control loss '''
    def init_aux_task_loss_pixel_control(self):
        self.pc_actions_taken = tf.placeholder(dtype=tf.float32, shape=[None,self.action_size])
        self.q_target = tf.placeholder(dtype=tf.float32,shape=[None, 20, 20])

        pc_actions_taken_reshaped = tf.reshape(self.pc_actions_taken,[-1,1,1,self.action_size])
        #Get Q value for action taken
        q_value = tf.multiply(self.pc_q, pc_actions_taken_reshaped)
        q_value = tf.reduce_sum(q_value, reduction_indices=3)

        #Log because the loss is very high
        #pc_loss = tf.reduce_sum(np.square(self.q_target - q_value))
        pc_loss = tf.reduce_sum(tf.nn.l2_loss(self.q_target - q_value))

        self.pc_loss = pc_loss * self.FLAGS.pc_loss_lambda

''' Wrapper for network'''
class NetworkWrapper:
    def __init__(self, scope, trainer,master_network, FLAGS):
        '''
            @scope : scope in which to create variables
            @trainer : trainer used to minimize loss
            @master_network : master network
            @FLAGS : specify different options through flags
        '''
        self.network_type = FLAGS.network_type
        self.network = None

        if self.network_type == 'LSTM':
            print ('LOG: Using lstm network')
            self.network = A3CLSTM(scope, trainer, master_network, FLAGS)
        elif self.network_type == 'FF':
            print ('LOG: FeedForward Network not implemented')
            raise NotImplementedError

    def get_network(self):
        return self.network
