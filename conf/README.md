# What is this for?

This folder stores configuration files used by Kedro.

## Configurations
the base folder contains the standard configurations, this is used when running a pipeline without specifying an environment. The 'emb' and 'ssd' contain alternative configurations. Specifically, the 'emb' configuration can be used to use precomputed embeddings, rather than using the encoder networks live. Note that no data augmentations can be used when using this configuration. The configuration assumes that the mebeddings are already stored on the ssd storage ('/data-gpu'), these can be downloaded to there using 'kedro run --pipepine download'.
The 'ssd' configurations can be used to train a model loading the audio and video files from the ssd. This supposes that the audio and video segments are stored on the ssd. 


