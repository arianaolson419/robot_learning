"""This script is used to slice the background noise into smaller clips and
save these clips into a new directory. These chunks are used to add background
noise to clips from the dataset.

Authors: Anna Buchele, Ariana Olson

Flags
-----
--base_path: the path from the home directory that contains unchunked background noise.
--dest_path: the path from the home directory to save the chunked background noise in
--chunk_length: The length in seconds to make each chunk of background noise.
"""
import scipy.io.wavfile as wav
import argparse
from os import environ
from os.path import join

HOME_DIR = environ["HOME"]

FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument(
        '--base_path',
        type=str,
        default='catkin_ws/src/robot_learning/AudioData/BackgroundNoise/unchunked/',
        help = "The path to the directory containing the background noise files.")
parser.add_argument(
        '--dest_path',
        type=str,
        default='catkin_ws/src/robot_learning/AudioData/BackgroundNoise/chunked/',
        help = "The path to the directory containing the chunked background noise files.")
parser.add_argument(
        '--chunk_length',
        type=float,
        default=1.0,
        help = "The length in seconds to make each chunk of background noise.")

noise_files = ['doing_the_dishes', 'dude_miaowing', 'exercise_bike', 'running_tap']

FLAGS, _ = parser.parse_known_args()

for f in noise_files:
    read_path = join(HOME_DIR, FLAGS.base_path + f + '.wav')
    rate, data = wav.read(read_path)
    size = int(FLAGS.chunk_length * rate) 
    start = 0
    end = size
    label = 0
    while end < len(data):
        write_path = join(HOME_DIR, FLAGS.dest_path + f + str(label) + '.wav')
        wav.write(write_path, rate, data[start:end])
        start += size
        end += size
        label += 1
