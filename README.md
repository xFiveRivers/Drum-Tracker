# **DrumTracker**

A command-line interface (CLI) drum loop MIDI transcription program.

## **About**

`DrumTracker` is a program that transcribes the timing of percussion hits for each instrument played in a drum loop through two core functions, `audio classification` and `midi transcription`. 

> Audio classification is handled through my convolutional neural network (CNN) classifier, found [here](https://github.com/xFiveRivers/drum-classifier).

## **Topics Explored**

* Creating an audio classification model using `PyTorch` and `TorchAudio`
* Exploratory data analysis (EDA) and audio signal data engineering with `TorchAudio`, `Librosa`, and `MatPlotLib`

## **Background**

Drum track and timing characteristics differ between genres of music. Electronic dance music (EDM) has typically has stiffly quantized percussion hits while jazz drummers have their own inherent swing and movement in their timing. My passion lies within hip-hop, specifically boom-bap with its analogue drum samples and organic swing from playing samples on a sampler.

As a music producer, I've always been interested in underlying patterns and characterisitics of classic drummers that hip-hop songs sampled. When making a beat, the drums are the foundation and backbone of the track from which the instruments ride on. When classic hip-hop producers lay down drums for their tracks, they typically took a drum loop (a section of a song where only the drums were being played) and segmented it into small chops for each instrument. One chop could be a hi-hat, another a kick, and another a snare. They would then load these samples into a sampler and play their own drum pattern using the samples extracted from the original loop.