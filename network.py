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


        #Construct network in current scope
        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            #Input
            self.input = tf.placeholder(dtype=tf.float32,shape=[None, self.input_size * self.input_size, self.no_frames],name='input')
            #Reshape input to be a 4d tensor
            self.input_reshaped = tf.reshape(self.input,shape=[-1, self.input_size, self.input_size, self.no_frames])

        #We need to reuse the CNN layers for auxiliary tasks
        #So the conv layers will take an arbitrary size input parameter
        self.cnn_output = self.cnn_trunck(self.input_reshaped)

        #Fully connected 256 layer
        self.fc_output = self.fc_trunck(self.cnn_output)

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
    def fc_trunck(self, input, reuse=False):
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
    def cnn_trunck(self, input, reuse=False):
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
            input_reshaped = tf.reshape(input, shape=[-1, 256])

            # VALUE
            shape = [256, 1]
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
            input_reshaped = tf.reshape(input, shape=[-1, 256])

            # POLICY
            shape = [256, 1 * self.action_size]
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

            #Add auxiliary tasks loss
            self.init_auxiliary_tasks_loss(self.FLAGS)

            #GRADIENTS COMPUTATION
            #Compute variabels norm (just for plotting needs)
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
            self.apply_grads = self.trainer.apply_gradients(zip(self.clipped_gradients, self.global_vari))


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
        cnn_output = self.cnn_trunck(prev_obs_reshaped, reuse=True)

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
            #The network up to this point is initialized by the BaseNetwork

            sequence_length = tf.shape(self.input_reshaped)[:1]

            # We need to initialize the lstm state in order to use it later
            self.lstm_state_init = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]), np.zeros([1, 256]))

            # LSTM
            self.lstm = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            # Create initial state for lstm
            self.initial_cell_c_state = tf.placeholder(dtype=tf.float32, shape=[1, 256], name='init_cell_c_state')
            self.initial_cell_h_state = tf.placeholder(dtype=tf.float32, shape=[1, 256], name='init_cell_h_state')
            self.lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_cell_c_state, self.initial_cell_h_state)

        rnn_outputs, self.lstm_state_out = self.lstm_trunck(self.fc_output, sequence_length, self.lstm_state)

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
    def lstm_trunck(self, input, sequence_length, lstm_state, reuse=False):
        with tf.device(self.device), tf.variable_scope(self.scope, reuse=reuse) as scope:
            # Reshape output to be ?, 256
            # 3d tensor to use with dynamic rnn which needs [batch_size, max_time, ...]
            input_reshaped = tf.reshape(input, shape=[1, -1, 256])

            # tf.nn.dynamic_rnn
            # time_major == False (default), this must be a Tensor of shape: [batch_size, max_time, ...]
            rnn_outputs, rnn_state = tf.nn.dynamic_rnn(self.lstm, input_reshaped,
                                                                 sequence_length=sequence_length,
                                                                 initial_state=lstm_state,
                                                                 time_major=False, scope=scope)

            lstm_state_out = (rnn_state[0][:1, :], rnn_state[1][:1, :])

            return rnn_outputs, lstm_state_out



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

        # We pass our frames through the CNN
        # Reuse variable
        cnn_output = self.cnn_trunck(prev_obs_reshaped, reuse=True)

        #We pass it through the FC layer
        fc_output = self.fc_trunck(cnn_output, reuse=True)

        #As opposed to the reward prediction, we also reuse the LSTM
        sequence_length = tf.shape(prev_obs_reshaped)[:1]

        #Initial state of the lstm should be 0?
        #Reset lstm state
        zero_state = self.lstm.zero_state(1, tf.float32)

        rnn_outputs, rnn_state = self.lstm_trunck(fc_output, sequence_length, zero_state, reuse=True)

        #Predict value
        self.vp_prediction = self.value_trunk(rnn_outputs, reuse=True)


    ''' Frame prediction network reuses the LSTM (with sampled experience)'''
    def init_aux_task_frame_prediction(self):
        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            self.previous_observations_fp = tf.placeholder(
                shape=[None, self.input_size * self.input_size, self.no_frames],
                dtype=tf.float32, name='previous_observations_fp')

            # Reshape to be a 4d tensor
            prev_obs_reshaped = tf.reshape(self.previous_observations_fp,
                                           shape=[-1, self.input_size, self.input_size, self.no_frames]
                                           ,
                                           name='previous_observations_fp_reshaped')

        # We pass our frames through the CNN
        # Reuse variable
        cnn_output = self.cnn_trunck(prev_obs_reshaped, reuse=True)

        # We pass it through the FC layer
        fc_output = self.fc_trunck(cnn_output, reuse=True)

        # As opposed to the reward prediction, we also reuse the LSTM
        sequence_length = tf.shape(prev_obs_reshaped)[:1]

        # Initial state of the lstm should be 0?
        # Reset lstm state
        zero_state = self.lstm.zero_state(1, tf.float32)

        rnn_outputs, rnn_state = self.lstm_trunck(fc_output, sequence_length, zero_state, reuse=True)
        rnn_outputs = tf.reshape(rnn_outputs, [-1, 256])

        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            print(rnn_outputs.shape)

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
        cnn_output = self.cnn_trunck(prev_obs_reshaped, reuse=True)

        # We pass it through the FC layer
        fc_output = self.fc_trunck(cnn_output, reuse=True)

        # As opposed to the reward prediction, we also reuse the LSTM
        sequence_length = tf.shape(prev_obs_reshaped)[:1]

        # Initial state of the lstm should be 0?
        # Reset lstm state
        zero_state = self.lstm.zero_state(1, tf.float32)

        rnn_outputs, rnn_state = self.lstm_trunck(fc_output, sequence_length, zero_state, reuse=True)
        rnn_outputs = tf.reshape(rnn_outputs, [-1, 256])

        with tf.device(self.device), tf.variable_scope(self.scope) as scope:
            # Map lstm output to 7x7x32
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
                out_height = (fc_pc.shape[1].value - 1) * 2 + W_deconv_value.shape[0].value
                out_width = (fc_pc.shape[2].value - 1) * 2 + W_deconv_value.shape[1].value
                out_shape = [tf.shape(fc_pc)[0], out_height, out_width,
                             W_deconv_value.shape[
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
                out_height = (fc_pc.shape[1].value - 1) * 2 + W_deconv_advantage.shape[0].value
                out_width = (fc_pc.shape[2].value - 1) * 2 + W_deconv_advantage.shape[1].value
                out_shape = [tf.shape(fc_pc)[0], out_height, out_width,
                             W_deconv_advantage.shape[2].value]
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
