import numpy as np
import os.path
import scipy.misc
try:
    import cPickle as pickle
except :
    import pickle
import tensorflow as tf
import time
from tensorflow.python.framework import graph_util
FLAGS = tf.app.flags.FLAGS

def _summarize_progress(train_data, feature, label, gene_output, batch, suffix, max_samples=8):
    td = train_data

    size = [label.shape[1], label.shape[2]]

    nearest = tf.image.resize_nearest_neighbor(feature, size)
    nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)

    bicubic = tf.image.resize_bicubic(feature, size)
    bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)

    clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

    # image   = tf.concat(2, [nearest, bicubic, clipped, label])
    image   = tf.concat([nearest, bicubic, clipped, label],2)

    image = image[0:max_samples,:,:,:]
    # image = tf.concat(0, [image[i,:,:,:] for i in range(max_samples)])
    image = tf.concat([image[i,:,:,:] for i in range(max_samples)],0)
    image = td.sess.run(image)
    # print('fea shape {}'.format(feature.shape)) # fea shape (16, 16, 16, 3)
    # print('label shape {}'.format(label.shape)) # label shape (16, 64, 64, 3)
    # print('gene_output shape {}'.format(gene_output.shape)) # gene_output shape (16, 64, 64, 3)

    filename = 'batch%06d_%s.png' % (batch, suffix)
    filename = os.path.join(FLAGS.train_dir, filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))

def _save_checkpoint(saver, train_data, batch):
    td = train_data
    newname = 'checkpoint_new'
    newname = os.path.join(FLAGS.checkpoint_dir, newname)
    # Generate new checkpoint
    saver.save(td.sess, newname,batch)
    output_graph_def = graph_util.convert_variables_to_constants(
            td.sess,
            td.sess.graph_def,
            ['gene_output']
        )
    pbname = '{}_{}_gene.pb'.format(newname,batch)
    with tf.gfile.GFile(pbname, "wb") as f: 
        f.write(output_graph_def.SerializeToString())

    print("    Checkpoint saved")
    return pbname

def train_model(train_data):
    td = train_data

    # summaries = tf.merge_all_summaries()
    summaries = tf.summary.merge_all()

    #td.sess.run(tf.initialize_all_variables())
    init = tf.global_variables_initializer()
    td.sess.run(init)
    saver = tf.train.Saver(max_to_keep=5 )
    print("init models using gloable: init")
    if FLAGS.load_epoch is not None:
        # resume from ckpt
        model_path = FLAGS.load_epoch
        ckpt = tf.train.get_checkpoint_state(model_path)
        print("restore model path:",model_path)
        readstate = ckpt and ckpt.model_checkpoint_path
        saver.restore(td.sess, ckpt.model_checkpoint_path)
        print("restore models' param")

    lrval       = FLAGS.learning_rate_start
    start_time  = time.time()
    done  = False
    batch = 0

    assert FLAGS.learning_rate_half_life % 10 == 0
    assert td.sess.graph is tf.get_default_graph()

    # Cache test features and labels (they are small)
    test_feature, test_label = td.sess.run([td.test_features, td.test_labels])
    # save for evaluate
    f = open('dump.fea', 'wb')
    pickle.dump(test_feature, f)
    print('test_feature shape {}'.format(test_feature.shape))
    f.close()
    while not done:
        batch += 1
        gene_loss = disc_real_loss = disc_fake_loss = -1.234

        feed_dict = {td.learning_rate : lrval}

        ops = [td.gene_minimize, td.disc_minimize, td.gene_loss, td.disc_real_loss, td.disc_fake_loss]
        _, _, gene_loss, disc_real_loss, disc_fake_loss = td.sess.run(ops, feed_dict=feed_dict)
        
        if batch % 10 == 0:
            # Show we are alive
            elapsed = int(time.time() - start_time)/60
            print('Progress[%3d%%], ETA[%4dm], Batch [%4d], G_Loss[%3.3f], D_Real_Loss[%3.3f], D_Fake_Loss[%3.3f]' %
                  (int(100*elapsed/FLAGS.train_time), FLAGS.train_time - elapsed,
                   batch, gene_loss, disc_real_loss, disc_fake_loss))

            # Finished?            
            current_progress = elapsed / FLAGS.train_time
            if current_progress >= 1.0:
                done = True
            
            # Update learning rate
            if batch % FLAGS.learning_rate_half_life == 0:
                lrval *= .5

        if batch % FLAGS.summary_period == 0:
            # Show progress with test features
            feed_dict = {td.gene_minput: test_feature}
            gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dict)
            _summarize_progress(td, test_feature, test_label, gene_output, batch, 'out')
            
        if batch % FLAGS.checkpoint_period == 0:
            # Save checkpoint
            pbname = _save_checkpoint(saver, td, batch)

            # config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            # detection_graph = tf.Graph()
            # with detection_graph.as_default():
            #     od_graph_def = tf.GraphDef()
            #     with tf.gfile.GFile(pbname, 'rb') as fid:
            #         serialized_graph = fid.read()
            #         od_graph_def.ParseFromString(serialized_graph)
            #         for node in od_graph_def.node:
            #             if node.op == 'RefSwitch':
            #                 node.op = 'Switch'
            #                 for index in range(len(node.input)):
            #                     if 'moving_' in node.input[index] and "Switch" not in node.input[index]:
            #                         node.input[index] = node.input[index] + '/read'
            #             elif node.op == 'AssignSub':
            #                 node.op = 'Sub'
            #                 if 'use_locking' in node.attr: del node.attr['use_locking']
            #             elif node.op == 'AssignAdd':
            #                 node.op = 'Add'
            #                 if 'use_locking' in node.attr: del node.attr['use_locking']
            #             elif node.op == 'Assign':
            #                 node.op = 'Identity'
            #                 if 'use_locking' in node.attr: del node.attr['use_locking']
            #                 if 'validate_shape' in node.attr: del node.attr['validate_shape']
            #                 if len(node.input) == 2:
            #                 # input0: ref: Should be from a Variable node. May be uninitialized.
            #                 # input1: value: The value to be assigned to the variable.
            #                     node.input[0] = node.input[1]
            #                     del node.input[1]

            #         tf.import_graph_def(od_graph_def, name='')

            # sess = tf.Session(graph=detection_graph, config=config)
            # image_tensor = sess.graph.get_tensor_by_name('inputs:0')
            # det_output = sess.graph.get_tensor_by_name('gene_output:0')
            # gene_output1 = sess.run(det_output, feed_dict={image_tensor:test_feature})
            # gene_output2 = td.sess.run(td.gene_moutput, feed_dict={td.gene_minput: test_feature})
            # diff = np.abs(gene_output1 - gene_output2)
            # # print('gene_output1 ', gene_output1)
            # # print('gene_output2 ', gene_output2)
            # # print('diff ', diff)
            # sess.close()

    _save_checkpoint(saver, td, batch)
    print('Finished training!')
