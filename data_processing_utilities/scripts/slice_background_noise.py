import scipy.io.wavfile as wav
import argparse

FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument(
        '--base_path',
        type=str,
        default='/home/ariana/catkin_ws/src/robot_learning/AudioData/BackgroundNoise/unchunked/',
        help = "The path to the directory containing the background noise files.")
parser.add_argument(
        '--dest_path',
        type=str,
        default='/home/ariana/catkin_ws/src/robot_learning/AudioData/BackgroundNoise/chunked/',
        help = "The path to the directory containing the chunked background noise files.")
parser.add_argument(
        '--chunk_length',
        type=float,
        default=1.0,
        help = "The length in seconds to make each chunk of background noise.")

noise_files = ['doing_the_dishes', 'dude_miaowing', 'exercise_bike', 'running_tap']

FLAGS, _ = parser.parse_known_args()

for f in noise_files:
    rate, data = wav.read(FLAGS.base_path + f + '.wav')
    size = int(FLAGS.chunk_length * rate) 
    start = 0
    end = size
    label = 0
    while end < len(data):
        print(f, size)
        wav.write(FLAGS.dest_path + f + str(label) + '.wav', rate, data[start:end])
        start += size
        end += size
        label += 1
