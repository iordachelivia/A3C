import os 
import threading
import multiprocessing
from helper import *
from network import NetworkWrapper
from worker import Worker
from config import *
from time import sleep, time


load_model = False
if load_model == True:
    FLAGS.experience_buffer_maxlen = 100
    FLAGS.episodes = 600

#Reset the graph
tf.reset_default_graph()


dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = dir_path + '/train_0'
frames_path = dir_path + '/frames'
#Create folders
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(frames_path):
    os.makedirs(frames_path)

# copy tsv for embeddings
bashCommand = "cp " + dir_path + "/embedding_metadata.tsv " + model_path
os.system(bashCommand)

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
with tf.device(FLAGS.device),tf.Session(config = config) as sess:
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes',
                                  trainable=False)
    trainer = tf.train.AdamOptimizer(1e-4,)
    #Create master network : it will hold the gradients
    #Each worker will update these gradients and sync with the master
    network_wrapepr = NetworkWrapper('global', trainer, None, FLAGS)
    master_network = network_wrapepr.get_network()

    # Set workers ot number of available CPU threads
    num_workers = multiprocessing.cpu_count()
    # num_workers = 1
    workers = []
    for index in range(num_workers):
        # Create worker classes
        worker = Worker(index, sess, trainer, dir_path, global_episodes,
                        master_network, FLAGS)
        # Initialize associated game
        worker.init_game(GAME_NAME, INPUT_SIZE)
        workers.append(worker)

    saver = tf.train.Saver(max_to_keep=20)

    coord = tf.train.Coordinator()
    if load_model == True:
        print ('LOG: Loading Model... %s'%model_path)
        # model_checkpoint_path = ckpt.model_checkpoint_path
        # model_checkpoint_path = model_path + '/model-500.ckpt'
        model_checkpoint_path = tf.train.latest_checkpoint(model_path)
        saver.restore(sess, model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    start = time()

    # Start the work process for each worker
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)

    end = time()
    minutes = (end - start)/60
    print('LOG: Training for %d episodes took %f minutes' % (FLAGS.episodes,
                                                             minutes))

'''
    From lab directory (where A3C directory was placed)
    COMMAND :bazel run :a3c_train --define headless=osmesa
    tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'
'''
