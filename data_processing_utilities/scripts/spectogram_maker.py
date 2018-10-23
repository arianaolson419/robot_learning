from scipy import signal, io, misc
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image


class SpectogramMaker(object):
	def __init__(self, path):
		self.input_audio_paths = [f for f in listdir(path) if isfile(join(path, f))]
		self.all_spectograms = []

	def make_spectogram(self, path):
		""" Makes a spectogram for the given path
		"""
		rate, data = io.wavfile.read(path)
		return signal.spectogram(data, rate)

	def make_all_spectograms(self):
		""" Makes a spectogram for each of the filepaths
		"""
		for path in self.input_audio_paths:
			self.all_spectograms.append([self.make_spectogram(path), os.path.basename(path)])

	def plot_spectogram(self, f, t, Sxx):
		""" Plots the given spectogram
		"""
		plt.pcolormesh(t, f, Sxx)
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		plt.show()
		return plt

	def save_spectograms(self, path):
		""" Saves all of the spectograms to the specified output path with the same filename as input
		"""
		for spectogram in self.all_spectograms:
			t, f, Sxx, basepath = spectogram
			output_path = join(path, basepath) + ".png"
			plt = self.plot_spectogram(f, t, Sxx)
			misc.imsave(output_path, plt)


	def make_and_show_dummy(self):
		""" For testing, generates a signal, turns it into a spectogram, and plots it
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
		plt = self.plot_spectogram(f, t, Sxx)
		misc.imsave("test.png", plt)


if __name__ == '__main__':
	spectogram_maker = SpectogramMaker("/")
	spectogram_maker.make_and_show_dummy()
