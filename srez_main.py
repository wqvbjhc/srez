'''
python srez_main.py --run train --dataset ./CelebA/Img/img_align_celeba --train_time 800
python srez_freeze_graph.py ./checkpoint ./good_model.pb
python srez_main.py --run evaluate  --pbfile ./good_model.pb
'''
import srez_demo
import srez_input
import srez_model
import srez_train

import os
import os.path
import random
import numpy as np
import numpy.random
import scipy.misc
import cv2
try:
    import cPickle as pickle
except :
    import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Configuration (alphabetically)
tf.app.flags.DEFINE_integer('batch_size', 16,
                            "Number of samples per batch.")

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                           "Output folder where checkpoints are dumped.")

tf.app.flags.DEFINE_integer('checkpoint_period', 10000,
                            "Number of batches in between checkpoints")

tf.app.flags.DEFINE_string('dataset', 'dataset',
                           "Path to the dataset directory.")

tf.app.flags.DEFINE_float('epsilon', 1e-8,
                          "Fuzz term to avoid numerical instability")

tf.app.flags.DEFINE_string('run', 'demo',
                            "Which operation to run. [demo|train]")

tf.app.flags.DEFINE_float('gene_l1_factor', .90,
                          "Multiplier for generator L1 loss term")

tf.app.flags.DEFINE_float('learning_beta1', 0.5,
                          "Beta1 parameter used for AdamOptimizer")

tf.app.flags.DEFINE_float('learning_rate_start', 0.00020,
                          "Starting learning rate used for AdamOptimizer")

tf.app.flags.DEFINE_integer('learning_rate_half_life', 5000,
                            "Number of batches until learning rate is halved")

tf.app.flags.DEFINE_bool('log_device_placement', False,
                         "Log the device where variables are placed.")

tf.app.flags.DEFINE_integer('sample_size', 64,
                            "Image sample size in pixels. Range [64,128]")

tf.app.flags.DEFINE_integer('summary_period', 200,
                            "Number of batches between summary data dumps")

tf.app.flags.DEFINE_integer('random_seed', 0,
                            "Seed used to initialize rng.")

tf.app.flags.DEFINE_integer('test_vectors', 16,
                            """Number of features to use for testing""")
                            
tf.app.flags.DEFINE_string('train_dir', 'train',
                           "Output folder where training logs are dumped.")

tf.app.flags.DEFINE_integer('train_time', 20,
                            "Time in minutes to train the model")

# add by wqvbjhc
tf.app.flags.DEFINE_string('load_epoch', None,
                            "the epoch path for resume")
tf.app.flags.DEFINE_string('pbfile', None,
                            "the pb file path for _evaluate")

def prepare_dirs(delete_train_dir=False):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
    # Cleanup train dir
    if delete_train_dir:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    # Return names of training files
    if not tf.gfile.Exists(FLAGS.dataset) or \
       not tf.gfile.IsDirectory(FLAGS.dataset):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.dataset,))

    filenames = tf.gfile.ListDirectory(FLAGS.dataset)
    filenames = sorted(filenames)
    random.shuffle(filenames)
    filenames = [os.path.join(FLAGS.dataset, f) for f in filenames]

    return filenames


def setup_tensorflow():
    # Create session
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement,gpu_options=gpu_options)
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
        
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    # summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    return sess, summary_writer

def _demo():
    # Load checkpoint
    if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.checkpoint_dir,))

    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    filenames = prepare_dirs(delete_train_dir=False)

    # Setup async input queues
    features, labels = srez_input.setup_inputs(sess, filenames, istrain=False)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_model.create_model(sess, features, labels)

    # Restore variables from checkpoint
    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt'
    filename = os.path.join(FLAGS.checkpoint_dir, filename)
    saver.restore(sess, filename)

    # Execute demo
    srez_demo.demo1(sess)

def _evaluate():
    # add by wqvbjhc - 20190701
    if not tf.gfile.Exists(FLAGS.pbfile):
        raise FileNotFoundError("Could not find pb file `%s'" % (FLAGS.pbfile,))

    f = open('dump.fea', 'rb')
    test_feature = pickle.load(f)
    f.close()
    print('test_feature shape {}'.format(test_feature.shape))

    # load model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FLAGS.pbfile, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph, config=config)
    image_tensor = sess.graph.get_tensor_by_name('inputs:0')
    det_output = sess.graph.get_tensor_by_name('gene_output:0')

    # 用其他图片测试，失败
    imglist=['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg',
             '6.jpg','7.jpg','8.jpg','9.jpg','10.jpg',
             '11.jpg','12.jpg','13.jpg','14.jpg','15.jpg',]
    imglist=['1.bmp','2.bmp','3.bmp','4.bmp','5.bmp',
             '6.bmp','7.bmp','8.bmp','9.bmp','10.bmp',
             '11.bmp','12.bmp','13.bmp','14.bmp','15.bmp',]
    for index, imgname in enumerate(imglist):
        img_output = cv2.imread(imgname)
        img_output = cv2.resize(img_output,(16,16))
        img_output = np.float32(img_output)
        img_output = img_output / 255.
        img_output = img_output[:,:,(2,1,0)]
        img_output = img_output[np.newaxis,:]

        # img_output = test_feature[index]
        # img_output = img_output[np.newaxis,:,:,:]
        # print(img_output.shape, img_output.dtype)
        name, ext = os.path.splitext(imgname)
        # filename = name+".bmp"
        # scipy.misc.toimage(np.squeeze(img_output), cmin=0., cmax=1.).save(filename)
        gene_output = sess.run(det_output, feed_dict={image_tensor: img_output})
        print(gene_output.shape)
        gene_output = np.clip(gene_output,0.,1.)
        filename = name+"_result"+ext
        # scipy.misc.toimage(np.squeeze(gene_output[0]), cmin=0., cmax=1.).save(filename) 
        scipy.misc.imsave(filename, np.squeeze(gene_output[0]))

    print("Finish EVALUATING")

class TrainData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def _train():
    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    all_filenames = prepare_dirs(delete_train_dir=True)

    # Separate training and test sets
    train_filenames = all_filenames[:-FLAGS.test_vectors]
    test_filenames  = all_filenames[-FLAGS.test_vectors:]

    # TBD: Maybe download dataset here

    # Setup async input queues
    train_features, train_labels = srez_input.setup_inputs(sess, train_filenames,istrain=True)
    test_features,  test_labels  = srez_input.setup_inputs(sess, test_filenames,istrain=False)

    # Add some noise during training (think denoising autoencoders)
    noise_level = .03
    noisy_train_features = train_features + \
                           tf.random_normal(train_features.get_shape(), stddev=noise_level)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_model.create_model(sess, noisy_train_features, train_labels)

    gene_loss = srez_model.create_generator_loss(disc_fake_output, gene_output, train_features)
    disc_real_loss, disc_fake_loss = \
                     srez_model.create_discriminator_loss(disc_real_output, disc_fake_output)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')
    
    (global_step, learning_rate, gene_minimize, disc_minimize) = \
            srez_model.create_optimizers(gene_loss, gene_var_list,
                                         disc_loss, disc_var_list)

    # Train model
    train_data = TrainData(locals())
    srez_train.train_model(train_data)

def main(argv=None):
    # Training or showing off?

    if FLAGS.run == 'demo':
        _demo()
    elif FLAGS.run == 'train':
        _train()
    elif FLAGS.run == 'evaluate':
        _evaluate()

if __name__ == '__main__':
  tf.app.run()
