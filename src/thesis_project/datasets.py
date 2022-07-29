from pathlib import Path, PurePosixPath
from typing import Tuple

import numpy as np
import pandas as pd
from kedro.io import AbstractDataSet
from scipy.io import wavfile
import torchaudio
import torch
import torchvision
import urllib
from pytorchvideo.data.encoded_video import EncodedVideo
from google.cloud import storage
import os
from sqlitedict import SqliteDict

""
# DataSet of audio wav files
class AudioDataSet(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = PurePosixPath(filepath)

    def _load(self) -> Tuple[int, np.ndarray]:
        return torchaudio.load(self._filepath, normalize=True)

    def _save(self, waveform: torch.Tensor):
        return torchaudio.save(self._filepath, waveform)

    # def _exists(self) -> bool:
    #     return Path(self._filepath.as_posix()).exists()

    def _describe(self):
        return dict(filepath=self._filepath)


# DataSet of audio wav files
class EmbeddingDataSet(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = filepath

    def _load(self) -> Tuple[int, np.ndarray]:
        return torch.load(self._filepath)

    def _save(self, data: torch.Tensor):
        return torch.save(data, self._filepath)

    # def _exists(self) -> bool:
    #     return Path(self._filepath.as_posix()).exists()

    def _describe(self):
        return dict(filepath=self._filepath)

    
client = storage.Client()
bucket = client.get_bucket('thesis_video_files')


# DataSet of audio wav files
class AudioDataSetBucket(AbstractDataSet):
    def __init__(self, filepath):
        #self._filepath = PurePosixPath(filepath)
        #self.client = storage.Client()
        #self.bucket = self.client.get_bucket('thesis_video_files')
        self._filepath = filepath[25:]

    def _load(self) -> Tuple[int, np.ndarray]:
        blob = bucket.get_blob(self._filepath)
        with open("tmp.wav", "wb") as file_obj:
            blob.download_to_file(file_obj)
        # load to tensor and apply augmentations
        audio_tensor = torchaudio.load('tmp.wav', normalize=True)
   
        os.remove('tmp.wav')

        return audio_tensor


    def _save(self, waveform: torch.Tensor):
        return torchaudio.save(self._filepath, waveform)

    # def _exists(self) -> bool:
    #     return Path(self._filepath.as_posix()).exists()

    def _describe(self):
        return dict(filepath=self._filepath)
    
    
# DataSet of audio wav files
class VideoDataSetBucket(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = filepath[25:]

    def _load(self) -> Tuple[int, np.ndarray]:
        blob = bucket.get_blob(self._filepath)
        with open("tmp.mp4", "wb") as file_obj:
            blob.download_to_file(file_obj)
        # load to tensor and apply augmentations
        encoded = EncodedVideo.from_path('tmp.mp4')  
        video_tensor = encoded.get_clip(start_sec=0, end_sec=encoded.duration)['video']
        os.remove('tmp.mp4')

        return video_tensor


    def _save(self, waveform: torch.Tensor):
        return torchaudio.save(self._filepath, waveform)

    # def _exists(self) -> bool:
    #     return Path(self._filepath.as_posix()).exists()

    def _describe(self):
        return dict(filepath=self._filepath)



# DataSet of vide0 mp4 files
class VideoDataSet(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = PurePosixPath(filepath)

    def _load(self):
        encoded =  EncodedVideo.from_path(self._filepath)
        video_tensor = encoded.get_clip(start_sec=0, end_sec=encoded.duration)['video']
        return video_tensor

    def _save(self, df: pd.DataFrame) -> None:
        # TODO: implement this
        pass


    # def _exists(self) -> bool:
    #     return Path(self._filepath.as_posix()).exists()

    def _describe(self):
        return dict(filepath=self._filepath)
