import librosa
import matplotlib.pyplot as plt
import numpy as np


class AudioFunk():
    def __init__(self):
        pass


    def plot_waveform(self, signal: np.ndarray, rate: int = 22050, 
                      title: str = 'Waveform', ax: np.ndarray|None = None):
        """Plots the waveform of a signal.

        Parameters
        ----------
        signal : np.ndarray
            Audio signal to plot.
        rate : int, optional
            Sample rate of signal.
        title : str, optional
            Title of the plot, by default 'Waveform'.
        ax : np.ndarray | None, optional
            Axes to plot on, by default None
        """

        time_axis = np.arange(start=0, stop=(signal.shape[0] / rate),)