# **DrumTracker**

A command-line drum loop MIDI transcription program.

## **About**

`DrumTracker` is a program that transcribes the timing of percussion hits for each instrument played in a drum loop through two core functions, `audio classification` and `midi transcription`. 

> Audio classification is handled through my convolutional neural network (CNN) classifier, found [here](https://github.com/xFiveRivers/drum-classifier).

## **Installation**

The `drum-tracker` environment needs to be installed in a few steps as `PyTorch` and it's components can cause conflicts and issues during installation.

1) Install the conda envrioment by running the following command in the root folder of the repository:

```
conda env create -f env.yml
```

2) Activate the environment:

```
conda activate drum-tracker
```

3) Install PyTorch and TorchAudio by following the instructions found [here](https://pytorch.org/get-started/locally/).

4) Install `torchsummary` and `torchmetrics` by running the following:

```
pip install torchsummary torchmetrics
```

## **Usage**

1) The pipeline works on the basis of short drum-loops (maximum 10-15 seconds). For added organization youu can place the drum-loop in the `data` folder.

2) Open a new terminal located at the root of the repository and activate the `drum-tracker` environment:

```
conda activate drum-traker
```

3) Run the program using the following command:

```
python src/get_midi.py
```

4) Enter the path to your drum loop. For example:

```
data/loop_01.wav
```

5) Enter the path (including filename) the location you want to save the MIDI file to. For example:

```
data/midi_files/test_midi.mid
```

6) The program will ask you if you know the tempo (beats per minute) of the drum-loop. If you do, type `yes` and hit enter. Then type the tempo of the drum-loop and hit enter again. If you don't know the tempo, type `no` and the program will try to determine the tempo.

7) The MIDI file will be saved at your specified location for you to use!

## **Limitations**

1) The classifier was only trained on old-school boom-bap type drums, specifically hi-hats, kicks, and snares. That is to say ther instruments (such as toms, rides, crashes, etc) will not be classified accurately and it will have a hard time with drum hits from other genres.

2) When the drum-loop has multiple instruments playing at the same time, it will only detect one. For example if a kick and hi-hat are played, only the kick will be detected and the hi-hat hit will not be in the MIDI file. This is a limitation that will be improved in future revisions.

3) The onset (peak) detection is not as accurate as it could be meaning that the timing for some instrument hits may be too early if there is a lot of sonic information in a short period of time.

## **Topics Explored**

* Creating an audio classification model using `PyTorch` and `TorchAudio`
* Exploratory data analysis (EDA) and audio signal data engineering with `TorchAudio`, `Librosa`, and `MatPlotLib`
* MIDI file transcription in `Pyhon` using `MIDIutil`

## **Background**

Drum track and timing characteristics differ between genres of music. Electronic dance music (EDM) has typically has stiffly quantized percussion hits while jazz drummers have their own inherent swing and movement in their timing. My passion lies within hip-hop, specifically boom-bap with its analogue drum samples and organic swing from playing samples on a sampler.

As a music producer, I've always been interested in underlying patterns and characterisitics of classic drummers that hip-hop songs sampled. When making a beat, the drums are the foundation and backbone of the track from which the instruments ride on. When classic hip-hop producers lay down drums for their tracks, they typically took a drum loop (a section of a song where only the drums were being played) and segmented it into small chops for each instrument. One chop could be a hi-hat, another a kick, and another a snare. They would then load these samples into a sampler and play their own drum pattern using the samples extracted from the original loop.