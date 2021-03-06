"""This script generates and saves spectrograms of clips from one speaker in the
dataset. The audio first has background noise added and the RMS loudness
normalized to a target -24dB

Authors: Anna Buchele, Ariana Olson

Usage
-----
* Change the speaker
    The spectrogram maker makes spectrograms for one directory at a time. To
    change the speaker, the path to the data and the destination path have to be
    changed at the bottom of this script.
"""
from scipy import signal, misc
import librosa
import librosa.display
import matplotlib.pyplot as plt
from os import listdir, environ
from os.path import isfile, join, basename
import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import spectrogram

from data_preprocess import data_preprocess

HOME_DIR = environ['HOME']


class SpectrogramMaker(object):
        def __init__(self, input_path, output_path, noise_path):
                self.input_audio_paths = [join(input_path, f) for f in listdir(input_path) if isfile(join(input_path, f))]
                self.output_path = output_path
                self.noise_path = noise_path
                self.all_spectrograms = []

        def make_spectrogram(self, path):
                """ Makes a spectrogram for the given path
                """
                print(path)
                data, rate = librosa.load(path, sr=16000, res_type='scipy')
                processed = data_preprocess(data, self.noise_path, length_s=0.5) 
                S = np.abs(librosa.stft(processed))
                return S

        def make_all_spectrograms(self):
                """ Makes a spectrogram for each of the filepaths
                """
                for path in self.input_audio_paths:
                        self.all_spectrograms.append([self.make_spectrogram(path), basename(path)])

        def plot_spectrogram(self, Sxx, path = "", show = False, save = False):
                """ Plots the given spectrogram, saves it to file
                """
                fig = plt.figure(frameon=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                pt = librosa.display.specshow(librosa.amplitude_to_db(Sxx, ref=np.max), y_axis='log', x_axis='time')
                if show:
                	plt.show()
                if save:
                        fig.savefig(path)

        def save_all_spectrograms(self, as_image = False):
                """ Saves all of the spectrograms to the specified output path with the same filename as input
                """
                print("save_all")
                for spectrogram in self.all_spectrograms:
                        D, basepath = spectrogram
                        output_path = join(self.output_path, basepath[:-4]) 
                       	if as_image:
                        	self.plot_spectrogram(D, output_path, True)
                        else:
                                np.save(output_path, D)


        def make_and_show_dummy(self):
                """ For testing, generates a signal, turns it into a spectrogram, and plots it
                """
                fs = 10e3
                N = 1e5
                amp = 2 * np.sqrt(2)
                noise_power = 0.01 * fs / 2
                time = np.arange(N) / float(fs)
                mod = 500*np.cos(2*np.pi*0.25*time)
                carrier = amp * np.sin(2*np.pi*3e3*time + mod)
                noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
                noise *= np.exp(-time/5)
                x = carrier + noise
                f, t, Sxx = signal.spectrogram(x, fs)
                plt = self.plot_spectrogram(f, t, Sxx)
                misc.imsave("test.jpg", plt)


if __name__ == '__main__':
        directory = "KL"
        spectrogram_maker = SpectrogramMaker(join(HOME_DIR, "catkin_ws/src/robot_learning/AudioData/chunked_data/" + directory), join(HOME_DIR, "catkin_ws/src/robot_learning/Spectrograms/" + directory), join(HOME_DIR, "catkin_ws/src/robot_learning/AudioData/BackgroundNoise/chunked"))
        spectrogram_maker.make_all_spectrograms()
        spectrogram_maker.save_all_spectrograms()
