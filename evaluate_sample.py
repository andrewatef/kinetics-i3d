# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import i3d

import random
import re
import os
import tempfile
import cv2
import argparse


import imageio
#import pyglet
from IPython import display
from moviepy.editor import *
# from urllib import request 

from preprocessing import load_video

_IMAGE_SIZE = 224

_SAMPLE_PATHS = {
    'rgb': 'data/VID_NaderRun_rgb.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, rgb600, flow, or joint',)
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')

tf.flags.DEFINE_string('video_path', 'data/VID_NaderRun.mp4', 'path to video',)

def main(args):
  

  if not os.path.exists(args.video_path):
        print("This file does not exist")
        return
  
  # sample all video from video_path at specified frame rate (FRAME_RATE param)
  sample_video = load_video(args.video_path,1000000000)	
  MyVideo_Path = args.video_path
  Video_duration=0
  # sample_video = np.expand_dims(sample_video, axis=0)

  video_name = args.video_path.split("/")[-1][:-4]
  npy_rgb_output = 'data/' + video_name + '_rgb.npy'
  np.save(npy_rgb_output, sample_video)


  _SAMPLE_PATHS = {
    'rgb': npy_rgb_output,
  }

  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = FLAGS.eval_type

  imagenet_pretrained = FLAGS.imagenet_pretrained

  NUM_CLASSES = 400

  if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

  kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

  if eval_type in ['rgb', 'rgb600', 'joint']:
    # RGB input has 3 channels.
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(None, None, _IMAGE_SIZE, _IMAGE_SIZE, 3))


    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      rgb_logits, _ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)


    rgb_variable_map = {}
    for variable in tf.global_variables():

      if variable.name.split('/')[0] == 'RGB':
        if eval_type == 'rgb600':
          rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
        else:
          rgb_variable_map[variable.name.replace(':0', '')] = variable

    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

 
  model_logits = rgb_logits 
  model_predictions = tf.nn.softmax(model_logits)
  
  #restore the model weights from the checkpoints
  with tf.Session() as sess:
    feed_dict = {}
    if eval_type in ['rgb', 'rgb600', 'joint']:
      if imagenet_pretrained:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
      else:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
      tf.logging.info('RGB checkpoint restored')
      rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
      tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
      
      
      #Convert long video to a list of short sample videos 
      #MyVideo_Path ="data/cricket.avi"  # Video path 
      sample_frame = 80
      Output_Videos =[]
      sample_list = []


      for x in range(int(rgb_sample.shape[0]/sample_frame)):
        sample_list.append(rgb_sample[x*sample_frame:(x+1)*sample_frame])

      if len(sample_list) == 0 :
        sample_list.append(rgb_sample)

      # Run the i3d model on the list of videos and print the top 5 actions of every video.
      # First add an empty dimension to the sample video as the model takes as input
      # a batch of videos.

      
	    My_Counter=0
      My_clip =VideoFileClip(MyVideo_Path)
      Video_duration = np.round(My_clip.duration,1) ##### by seconds 
      Max_video_duration = np.round(Video_duration/len(sample_list),1)
      My_start=0 
      My_End=Max_video_duration

      ######  intial values for cutting the long video to short videos 

      
      for x in sample_list:
        model_input = np.expand_dims(x, axis=0)
        feed_dict[rgb_input] = model_input

        out_logits, out_predictions = sess.run([model_logits, model_predictions],
        feed_dict=feed_dict)

        out_logits = out_logits[0]
        out_predictions = out_predictions[0]
        sorted_indices = np.argsort(out_predictions)[::-1]

          #print('Norm of logits: %f' % np.linalg.norm(out_logits))
          #print('\nTop 5 classes and probabilities')
        My_count=0
        for index in sorted_indices[:5]:
            #print("%-22s %.2f%%" % (kinetics_classes[index], out_predictions[index] * 100))
            if My_count ==0 :
              result_text = kinetics_classes[index]
              My_count = My_count+1
			        break
        
        ###### Adding the text of the video action at the bottom .. 

        clip = VideoFileClip(MyVideo_Path).subclip(My_start,My_End)
        txt_clip = ( TextClip(result_text,fontsize=50,color='white')
              .set_position('bottom')
              .set_duration(Max_video_duration)
                )
        if My_start!=0 and My_start>=Video_duration:
           break 
        if My_start!=0 and My_End>=Video_duration:
            My_End = Video_duration
        video = CompositeVideoClip([clip, txt_clip])
        OutPut_name = str(My_Counter)+".mp4"
        video.write_videofile(OutPut_name,fps=25)
        Output_Videos.append(VideoFileClip(OutPut_name))
        clip.close()
        My_Counter=My_Counter+1
        My_start = My_start +Max_video_duration
        My_End = My_End +Max_video_duration

 
# concatenate all the results videos 
      Result_Video = concatenate_videoclips(Output_Videos ,method='compose')
      Result_Video.write_videofile("output_videos/Multi_10.mp4",fps=25) 
  





          
 


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--video_path', type=str, default="data/VID_NaderRun.mp4")

  args = parser.parse_args()
  main(args)
  #tf.compat.v1.app.run (main(args))

  