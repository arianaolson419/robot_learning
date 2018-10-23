from scipy import signal, misc
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, basename
import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import spectrogram


class SpectrogramMaker(object):
	def __init__(self, input_path, output_path):
		self.input_audio_paths = [join(input_path, f) for f in listdir(input_path) if isfile(join(input_path, f))]
		self.output_path = output_path
		self.all_spectrograms = []

	def make_spectrogram(self, path):
		""" Makes a spectrogram for the given path
		"""
		rate, data = wavfile.read(path)
		return spectrogram(data, rate)

	def make_all_spectrograms(self):
		""" Makes a spectrogram for each of the filepaths
		"""
		for path in self.input_audio_paths:
			self.all_spectrograms.append([self.make_spectrogram(path), basename(path)])

	def plot_spectrogram(self, f, t, Sxx):
		""" Plots the given spectrogram
		"""
		plt.pcolormesh(t, f, Sxx)
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		plt.show()
		return plt

	def save_spectrograms(self):
		""" Saves all of the spectrograms to the specified output path with the same filename as input
		"""
		for spectrogram in self.all_spectrograms:
			t, f, Sxx, basepath = spectrogram
			output_path = join(self.output_path, basepath) + ".png"
			plt = self.plot_spectrogram(f, t, Sxx)
			misc.imsave(output_path, plt)


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
		misc.imsave("test.png", plt)


if __name__ == '__main__':
	spectrogram_maker = SpectrogramMaker("AudioData/DC", "spectrograms/Data")
	spectrogram_maker.make_all_spectrograms()
	spectrogram_maker.save_spectrograms()
