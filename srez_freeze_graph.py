# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
"""
export pb model for inference
python srez_freeze_graph.py ./checkpoint ./good_model.pb
"""

import os
import sys
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(path))
sys.path[0], sys.path[-1] = sys.path[-1], sys.path[0]
print(sys.path)
import numpy as np
import srez_model
import tensorflow as tf
from tensorflow.python.framework import graph_util
import argparse
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    # meta_files = [s for s in files if s.endswith('.meta')]
    # if len(meta_files) == 0:
    #     raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    # elif len(meta_files) > 1:
    #     raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    # meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        meta_file = ckpt_file + '.meta'
        return meta_file, ckpt_file

    meta_files = [s for s in files if s.endswith('.meta')]
    max_step = -1
    for f in meta_files:
        step = int(f.split('.')[0].split('_')[-1])
        if step > max_step:
            max_step = step
            meta_file = f
            ckpt_file = f.split('.')[0] + '.ckpt'
    return meta_file, ckpt_file

def construct_model(sess):
    test_labels = np.random.rand(16,64,64,3)
    test_features = tf.placeholder(tf.float32, shape=[16,16,16, 3])
    test_labels = tf.placeholder(tf.float32, shape=[16,64,64, 3])
    in_tensor, out_tensor = srez_model.create_inference_model(sess,test_features,  test_labels)

def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model metagraph and checkpoint
            print('Model directory: %s' % args.model_dir)
            model_dir_exp = os.path.expanduser(args.model_dir)
            meta_file, ckpt_file = get_model_filenames(os.path.expanduser(args.model_dir))
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            construct_model(sess)
            # sess.run(tf.global_variables_initializer())
            # sess.run(tf.local_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(model_dir_exp, ckpt_file))
            
            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()
            
            # Freeze the graph def
            output_graph_def = freeze_graph_def(sess, input_graph_def, 'gene_output')

        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(args.output_file, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph: %s" % (len(output_graph_def.node), args.output_file))
        
def freeze_graph_def(sess, input_graph_def, output_node_names):
    '''
    Error when loading the frozen graph with tensorflow.contrib.layers.python.layers.batch_norm
    ValueError: graph_def is invalid at node u'BatchNorm/cond/AssignMovingAvg/Switch': Input tensor 'BatchNorm/moving_mean:0' Cannot convert a tensor of type float32 to an input of type float32_ref
    freeze_graph.py doesn't seem to store moving_mean and moving_variance properly

    An ugly way to get it working:
    manually replace the wrong node definitions in the frozen graph
    RefSwitch --> Switch + add '/read' to the input names
    AssignSub --> Sub + remove use_locking attributes
    '''
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index] and "Switch" not in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'Assign':
            node.op = 'Identity'
            if 'use_locking' in node.attr: del node.attr['use_locking']
            if 'validate_shape' in node.attr: del node.attr['validate_shape']
            if len(node.input) == 2:
                # input0: ref: Should be from a Variable node. May be uninitialized.
                # input1: value: The value to be assigned to the variable.
                node.input[0] = node.input[1]
                del node.input[1]

    # Get the list of important nodes
    whitelist_names = []
    for node in input_graph_def.node:
        if (node.name.startswith('MobileFaceNet') or node.name.startswith('embeddings')):
            whitelist_names.append(node.name)

    # Replace all the variables in the graph with constants of the same values
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","),
        variable_names_whitelist=None)
    return output_graph_def
  
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model_dir', type=str, 
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('output_file', type=str, 
        help='Filename for the exported graphdef protobuf (.pb)')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
