import csv
import librosa
import os
import pandas as pd
import re
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


class Preprocessor():
    def __init__(self, input_freq: int = 44100, resample_freq: int = 16000):

        self.input_freq = input_freq
        self.resample_freq = resample_freq

        self.resample = T.Resample(
            orig_freq = self.input_freq, 
            new_freq = self.resample_freq
        )

        N_FFT = 256
        HOP_LEN = N_FFT // 8
        N_MELS = 256

        self.mel_spec_trans = T.MelSpectrogram(
            sample_rate = self.resample_freq,
            n_fft = N_FFT,
            hop_length = HOP_LEN,
            n_mels = N_MELS
        )


    def forward(self, chops: list) -> list:
        processed_chops = []

        for chop in chops:
            # Transform to mono then downsample
            chop = self.resample(self._stereo_to_mono(chop))
            # Get mel-spectrogram
            chop = self.mel_spec_trans(chop)
            # Append to list
            processed_chops.append(chop)

        return processed_chops



    def _stereo_to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Converts a stereo waveform to mono

        Parameters
        ----------
        waveform : torch.Tensor
            Waveform to convert to mono.

        Returns
        -------
        torch.Tensor
            Waveform converted to mono.
        """

        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            return(torch.mean(waveform, dim=0, keepdim=True))
        
    