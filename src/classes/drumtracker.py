from midiutil import MIDIFile
from models import Model_02
import librosa
import numpy as np
import torch
import torchaudio.transforms as T

import warnings
warnings.filterwarnings('ignore')


class DrumTracker():
    def __init__(self):
        # Initialize model
        self._init_model()

        # Define class map for prediction and midi mapping
        self.class_map = {
            0 : ['snare', 71],
            1 : ['hat', 72],
            2 : ['kick', 70]
        }
    

    def get_midi(self):
        """Processes drum loop to extract MIDI.
        """

        # Get loop file location from user
        loop_path = input('Enter the path to the drum loop: ')

        # Load in drum loop
        self.loop, self.loop_rate = librosa.load(loop_path, sr=None)

        # Initialize functions
        self._init_funcs()

        # Extract chops
        self.oenv, onsets = self._detect_onsets()
        self.chop_dict = self._extract_chops(onsets)

        # Get prediction of each chop
        for chop in self.chop_dict['chops']:
            map_value = self.class_map[self._predict_chop(chop)]
            self.chop_dict['labels'].append(map_value[0])
            self.chop_dict['notes'].append(map_value[1])
        
        # Create midi transcription and file
        self._transcribe_midi()


    def _init_model(self):
        """Initializes classifier model
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model_02()
        if device == 'cuda':
            self.model.load_state_dict(torch.load('src/models/model_02.pt'))
        else:
            self.model.load_state_dict(torch.load(
                'src/models/model_02.pt',
                map_location=torch.device('cpu')
        ))
            
    
    def _init_funcs(self):
        """Initializes TorchAudio Transform Functions
        """

        self.power_to_db = T.AmplitudeToDB(
            stype = "amplitude",
            top_db = 80
        )
        self.resample = T.Resample(
            orig_freq = self.loop_rate,
            new_freq = 16000
        )
        self.mel_spec_trans = T.MelSpectrogram(
            sample_rate = 16000,
            n_fft = 256,
            hop_length = 256 // 8,
            n_mels = 256
        )
        self.midi = MIDIFile(1)


    def _detect_onsets(self) -> tuple[np.ndarray, np.ndarray]:
        """Detects onset peaks in a sample loop.

        Returns
        -------
        oenv : np.ndarray
            Onset strength envelope.
        onsets : np.ndarray
            Locations of onset peaks in terms of samples.
        """

        # Get onset strength envelope
        oenv = librosa.onset.onset_strength(
            y = self.loop,
            sr = self.loop_rate,
            hop_length = 1024
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
        """Extracts instrument chops from a sample loop.

        Parameters
        ----------
        onsets : list
            Location of onset peaks in terms of samples.

        Returns
        -------
        dict
            Properties of the chops.
        """

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
    

    def _predict_chop(self, chop: torch.Tensor) -> int:
        """Predicts class of chop

        Parameters
        ----------
        chop : torch.Tensor
            Chop of loop sample to predict.

        Returns
        -------
        int
            Prediction result from model.
        """

        preds = np.empty((0, 3), float)
        chunks = self._segment(self._to_mono(self.resample(torch.tensor(chop))))
        self.model.eval()
        with torch.no_grad():
            for chunk in chunks[:8]:
                if self._get_avg_db(chunk) > -80:
                    logits = self.model(
                        self.mel_spec_trans(chunk).unsqueeze(0).unsqueeze(0)
                    )
                    preds = np.append(
                        preds,
                        torch.softmax(logits, dim=1).numpy(),
                        axis = 0
                    )

        return np.mean(preds, axis=0).argmax()


    def _to_mono(self, signal: torch.Tensor) -> torch.Tensor:
        """Converts multi-channel audio to mono.

        Parameters
        ----------
        signal : torch.Tensor
            Audio signal to convert.

        Returns
        -------
        torch.Tensor
            Mono audio signal.
        """

        if len(signal.shape) > 1 and signal.shape[0] > 1:
                return(torch.mean(signal, dim=0, keepdim=True))
        else:
            return signal
        
    
    def _segment(self, signal: torch.Tensor, chunk_length: int = 160) -> list:
        """Segments audio into chunks.

        Parameters
        ----------
        signal : torch.Tensor
            Audio signal to segment.
        chunk_length : int, optional
            Length of chunks in samples, by default 160

        Returns
        -------
        list
            List of segmented chunks.
        """

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
    

    def _transcribe_midi(self):
        """Transcribes and creates a MIDI file.
        """
        
        file_path = input('Enter the desired path of the ouptut midi file: ')

        if input('Do you know the tempo (bpm) of the loop? [yes/no]: ') == 'yes':
            bpm = int(input('Please enter the bpm: '))
        else:
            bpm = int(librosa.beat.tempo(
                y = self.loop,
                sr = self.loop_rate,
                onset_envelope = self.oenv,
                start_bpm = 85
            )[0])
            print(f'Tempo autodetected as {bpm}bpm.')

        track, channel, time, duration, volume = 0, 0, 0, 0.25, 100
        self.midi.addTempo(track, time, bpm)

        for i, note in enumerate(self.chop_dict['notes']):
            q_time = (1 / 60) * bpm * self.chop_dict['times'][i]
            self.midi.addNote(track, channel, note, q_time, duration, volume)

        with open(file_path, 'wb') as out_file:
            self.midi.writeFile(out_file)

        print(f'Midi file saved to {file_path}.')