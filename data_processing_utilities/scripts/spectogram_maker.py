from scipy import signal, io
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np


class SpectogramMaker(object):
	def __init__(self, path):
		self.input_audio_paths = [f for f in listdir(path) if isfile(join(path, f))]
		self.all_spectograms = []

	def make_spectogram(self, path):
		rate, data = io.wavfile.read(path)
		return signal.spectogram(data, rate)

	def plot_spectogram(self, f, t, Sxx):
		plt.pcolormesh(t, f, Sxx)
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		plt.show()

	def make_and_show_dummy(self):
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
		self.plot_spectogram(f, t, Sxx)

if __name__ == '__main__':
	spectogram_maker = SpectogramMaker("/")
	spectogram_maker.make_and_show_dummy()
