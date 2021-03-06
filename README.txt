    _             _ _       _     _        ____        _        
   / \__   ____ _(_) | __ _| |__ | | ___  |  _ \  __ _| |_ __ _ 
  / _ \ \ / / _` | | |/ _` | '_ \| |/ _ \ | | | |/ _` | __/ _` |
 / ___ \ V / (_| | | | (_| | |_) | |  __/ | |_| | (_| | || (_| |
/_/   \_\_/ \__,_|_|_|\__,_|_.__/|_|\___| |____/ \__,_|\__\__,_|
                                                                
AM data contains 35M sorted spikes, and 53M unsorted spikes (from a subset of sessions) 
(sorted and unsorted are partially overlapping sets). 
Pooling the lab, will likely have on the order of 100M waveforms and a 
few 100G of data (if we also include spiketimes).

 ____                 _       _                     _ 
/ ___|  ___ _ __ __ _| |_ ___| |__  _ __   __ _  __| |
\___ \ / __| '__/ _` | __/ __| '_ \| '_ \ / _` |/ _` |
 ___) | (__| | | (_| | || (__| | | | |_) | (_| | (_| |
|____/ \___|_|  \__,_|\__\___|_| |_| .__/ \__,_|\__,_|
                                   |_|                

Challenges:
- Aggregating data across the lab
- Dealing with waveforms containing non-uniform lengths and registering waveforms
- Getting labelled data for "not spikes"
    - Approches:
        - Adding during spike sorting
        - Using semi-supervised methods to iteratively label more data
- Handling data efficiently
- Defining features to be used for clustering
    1) PCA of waveforms
    2) UMAP of waveforms
    3) Dot product with templates of neuron shapes
    4) (To address uneven waveform length) Maximum of cross-convolution with templates
- Testing different clustering algorithms.

Method of operation:
    - Classifier will be trained and stored in central location
    - Each time a user initiates sorting, classifier will be checked for updates
        and downloaded as needed, then run on the local computer

Metrics (what are we trying to optimize):
    - Classifier should identify as many true positives as possible, even
        if rate of false positives is high (don't want to miss spikes)
    - We also want to throw out negatives very conservatively (don't want
        to accidentally throw out a real neuron)

Stages:
    1) Full-batch classifier
    2) (Potentially) Online training

Data usage:
    - Given high redundancy in spike shapes (and number of waveforms per neuron),
        do we want to use all waveforms? 
    - Can we intelligently subset the data to make training faster?
    - Can we generate templates from neurons and use that instead of raw waveforms
