from midiutil import MIDIFile
from models import Model_02
import librosa
import numpy as np
import torch
import torchaudio.transforms as T


class DrumTracker():
    def __init__(self, loop_path: str):
        self.loop, self.loop_rate = librosa.load(loop_path, sr=None)

        self.class_mapping = {
            0 : ['snare', 71],
            1 : ['hat', 70],
            2 : ['kick', 72]
        }


    def _model_init(self):
        


    def _get_midi(self):
        oenv, onsets = self._detect_onsets()
        self.chop_dict = self._extract_chops(onsets)

    

    def _detect_onsets(self):
        # Get onset strength envelope
        oenv = librosa.onset.onset_strength(
            y = self.loop,
            sr = self.loop_rate, hop_legnth=1024
        )

        # Get onset frames
        onsets = librosa.onset.onset_detect(
            onset_envelope = oenv,
            backtrack = False
        )
        onsets = np.insert(onsets, 0, 0)
        onsets = librosa.frames_to_samples(onsets, hop_length=1024)
        onsets = np.append(onsets, len(self.loop))

        return oenv, onsets
    

    def _extract_chops(self, onsets: list) -> dict:
        chop_dict = {
            'chops': [],
            'lengths': [],
            'times': [],
            'labels': [],
            'notes': []
        }

        # Iterate through onsets in sample
        for i in range(len(onsets) - 1):
            # Segment chop from sample based on onset sample markers
            chop = self.loop[onsets[i] : onsets[i + 1]]

            # Find zero-cross points in chop
            zero_cross = np.nonzero(librosa.zero_crossings(chop))[0]

            # Segment chop to first and last detected zero-cross point
            chop = chop[zero_cross[0]:zero_cross[-1]]

            # Append values to dict
            chop_dict['chops'].append(chop)
            chop_dict['lengths'].append(len(chop) / self.loop_rate)
            chop_dict['times'].append(onsets[i] / self.loop_rate)

        return chop_dict
    

    def _process_chops(self):
        resample = T.Resample(orig_freq = self.loop_rate, new_freq = 16000)


    def _stereo_to_mono(self, signal: torch.Tensor) -> torch.Tensor:
        if len(signal.shape) > 1 and signal.shape[0] > 1:
                return(torch.mean(signal, dim=0, keepdim=True))
        else:
            return signal
        
    
    def _segment(self, signal: torch.Tensor, chunk_length: int = 160) -> list:
        chunks = torch.split(signal, chunk_length)

        return chunks
    

    def _get_avg_db(self, signal: torch.Tensor) -> float:
        """Returns the average decibel level of a signal.

        Parameters
        ----------
        signal : torch.Tensor
            Signal in power amplitude domain.

        Returns
        -------
        float
            Average decibel level of signal.
        """
        power_to_db = T.AmplitudeToDB(stype="amplitude", top_db=80)

        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0)

        signal_db = power_to_db(signal)

        return(round(float(torch.mean(signal_db.squeeze(0))), 4))
    

    def _mel_spec_trans(self, signal: torch.Tensor):
        # Hyperparameters for mel spectrogram transformation
        N_FFT = 256
        HOP_LEN = N_FFT // 8
        N_MELS = 256

        # PyTorch transform function for mel spectrogram
        mel_spec_trans = T.MelSpectrogram(
            sample_rate = 16000,
            n_fft = N_FFT,
            hop_length = HOP_LEN,
            n_mels = N_MELS
        )