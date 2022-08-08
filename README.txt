                  ____                                                   _ 
 _ __   ___ _   _|  _ \ ___  ___ ___  _ __ ___  _ __ ___   ___ _ __   __| |
| '_ \ / _ \ | | | |_) / _ \/ __/ _ \| '_ ` _ \| '_ ` _ \ / _ \ '_ \ / _` |
| | | |  __/ |_| |  _ <  __/ (_| (_) | | | | | | | | | | |  __/ | | | (_| |
|_| |_|\___|\__,_|_| \_\___|\___\___/|_| |_| |_|_| |_| |_|\___|_| |_|\__,_|
                                                                           
Spike waveform classifier aimed at:
    1- Removing noise during preprocessing for improved clustering
        1.5- Output from classifier provides an additional high
             quality feature for clustering on
    2- Recommending electrodes with neurons for user ease

== Note:
    - Classifier threshold is internationally set conservatively
        so false negatives are minimized (i.e. number of actual spikes 
        discarded is minimized).

############################################################
## Dataset
############################################################
Aiming for a 2GB dataset, half and half for spikes and noise.

== Suggestion for updating dataset
Add maximum diversity of both noise and spike clusters (i.e. if we have a 
total of 2000 neurons, and space for 100,000 waveforms, then every neuron
should contribute 100,000/2000 waveforms if possible)

############################################################
## Integration
############################################################
Aimed at being integrated into blech_clust/blech_process.py, prior
to actually performing clustering

== Method of operation:
    - Classifier will be trained and stored in central location
    - Each time a user initiates sorting, classifier will be checked for updates
        and downloaded as needed, then run on the local computer

############################################################
## Future Challenges
############################################################
- Dealing with waveforms of different lengths
    1- Either have multiple models,
    2- Or standardize waveform length
    ** Best waveform snapshot can be empirically determined
 
- Getting more labelled data for "not spikes"
    - Approches:
        - Adding during spike sorting
        - Using semi-supervised methods to iteratively label more data

############################################################
## Experimentation, Model Registry, Data Availability
############################################################
Experimentation and model + data tracking is done on Neptune.ai.
These details are not currently available publicly.
Dataset used here can be accessed at
https://drive.google.com/drive/folders/1i1WPL7gt0ckvpuGVoZKfnu27bRRVUQEX?usp=sharing
