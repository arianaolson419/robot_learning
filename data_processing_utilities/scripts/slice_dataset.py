import scipy.io.wavfile as wav
import numpy as np
import argparse
from os.path import join
from os import environ, listdir

HOME_DIR = environ['HOME']
base_dir = join(HOME_DIR, 'catkin_ws/src/robot_learning/AudioData')
base_dest = join(base_dir, 'chunked_data')

FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument(
        '--chunk_length',
        type=float,
        default=0.5,
        help = "The length in seconds to make each chunk of background noise.")

dataset_directories = ['DC', 'JE', 'JK', 'KL']

FLAGS, _ = parser.parse_known_args()

for d in dataset_directories:
    d_path = join(base_dir, d)
    for f in listdir(d_path):
        rate, data = wav.read(join(base_dir, d, f))
        size = int(FLAGS.chunk_length * rate)
        start = 0
        num_chunks = int(np.floor(data.shape[0] / size))
        for i in range(num_chunks):
            label = '{}{}.wav'.format(f.split('.')[0], i)
            print(label, size)
            chunk = data[i * size: i * size + size]
            print(data.shape, chunk.shape)
            wav.write(join(base_dest, d, label), rate, chunk) 
