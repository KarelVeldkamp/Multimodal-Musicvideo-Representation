from logging import exception
import numpy as np
import random
import re
import os
import torch
import torchaudio_augmentations.augmentations as ta
import torchvision
from torch.utils.data.sampler import Sampler
from pytorchvideo.transforms import RandomResizedCrop, Normalize
from torchvision import transforms
from torch.utils.data.sampler import Sampler, BatchSampler
from torch.utils.data import DataLoader, Dataset
from torchaudio_augmentations.apply import RandomApply
from typing import Dict
import warnings
from sqlitedict import SqliteDict
from tqdm import tqdm
import pandas
from sklearn import preprocessing

class MultiModalDataset(Dataset):
    """
    custom dataset compatible with pytorch for audio and video data in google bucket
    """
    def __init__(self, 
                 audio_transforms: torchvision.transforms.transforms.Compose, 
                 video_transforms: torchvision.transforms.transforms.Compose, 
                 n_audio_samples: int, 
                 n_video_samples: int,
                 audio_loaders: Dict[str, callable],
                 video_loaders: Dict[str, callable],
                 n:int=None):
        """
        bucket_name: name of google cloud storage bucket to load data from
        audio_transforms: augmentations to perform on the audio track
        video_transforms: augmentations to perform on the video
        n_audio_samples: number of audio segments to use per segment
        n_video_samples: number of video segments to use per segment
        audio_loaders: kedro lazy data loaders for audio files
        video_loaders: kedro lazy data loaders for video files
        """
        # save function parameters to attributes
        self.audio_transforms = audio_transforms
        self.video_transforms = video_transforms
        self.n_video_samples = n_video_samples
        self.n_audio_samples = n_audio_samples
        self.clipaudio = ClipAudio(n_audio_samples)
        self.clipvideo = ClipVideo(n_video_samples)
        self.audio_loaders = audio_loaders
        self.video_loaders = video_loaders
        
        # check that all segments have both an mp4 and a wav file
        if len(self.audio_loaders) != len(self.video_loaders):
            raise Exception("Some segments don't have both an audio and a video file") 
            
        # get list of segments
        segments = list(self.audio_loaders.keys())
        
        # get list of music video ids (with duplicates) and create a dict that maps the ids to their number of segments
        music_videos = []
        for segment in tqdm(segments):
            music_videos.append(segment.split('_')[0])
        self.segments_per_video = dict()
        for mv in tqdm(np.unique(music_videos)):
            self.segments_per_video[mv] = music_videos.count(mv)
            
        # only save a list of unique music video ids as attribute(without duplicates)
        self.music_videos = np.unique(music_videos)
        
        # if n is specified, only sample from n videos
        if n:
            self.music_videos = self.music_videos[:n]
        
        # a dictionary that maps an index to a key
        self.idxdict = {name:index for name, index in zip(self.music_videos, range(len(self.music_videos)))}

    def __len__(self):
        """
        return number of audio segments in the dataset (equal to number of video segments)
        """
        return len(self.music_videos)

    def __getitem__(self, 
                    idx: str):
        """
        idx:     xite music video id
        returns: audio and video tensors for a random segment of that video. Also includes an index  
        """
        # select random segment and download audio and video by calling kedro loader
        n_segments = self.segments_per_video[idx]
        segment_name = f'{idx}_{random.randint(0, n_segments-1)}'
        video_tensor = self.video_loaders[segment_name]()
        audio_tensor, _ = self.audio_loaders[segment_name]()
        
        # TODO: do this in ffmpeg not when loading data
        audio_tensor = audio_tensor[0, :]

        # get the latest possible start of the segment (as 0<float<1), and sample a starting point
        latest_possible_start = 1 - (self.n_video_samples/video_tensor.shape[1])
        segment_start = np.random.uniform(0, latest_possible_start)
        
        # cut out synchronous parts from audio and video
        audio_tensor = self.clipaudio(audio_tensor, segment_start)
        video_tensor = self.clipvideo(video_tensor, segment_start)
        
        # apply augmentations to video and audio
        video_tensor = self.video_transforms(video_tensor)
        audio_tensor = self.audio_transforms(audio_tensor[None, None, :])[0, 0, :]

        return audio_tensor, video_tensor, self.idxdict[idx]

class EmbeddingDataset(Dataset):
    """
    custom dataset compatible with pytorch for audio and video embeddings on the SSD. 
    Only loads the fist segemnt of each music video
    """
    def __init__(self, 
                 audio_loaders: Dict,
                 video_loaders: Dict,
                 n:int=None):
        """
        audio_loaders: kedro lazy data loaders for audio embeddings
        video_loaders: kedro lazy data loaders for video embeddings
        """
        # save function parameters to attributes
        self.audio_loaders = audio_loaders
        self.video_loaders = video_loaders
        
        # check that all segments have both an mp4 and a wav file
        if set(self.audio_loaders.keys()) != set(self.video_loaders.keys()):
            raise Exception("Some segments don't have both an audio and a video file") 
            
        # get list of segments
        segments = self.audio_loaders.keys()
        # get list of music video ids (with duplicates) and create a dict that maps the ids to their number of segments
        music_videos = []
        print('Finding unique music videos')
        for segment in tqdm(segments):
            music_videos.append(segment.split('_')[0])
        self.segments_per_video = dict()
        print('Calculating number of segments per music videos')
        for mv in tqdm(np.unique(music_videos)):
            self.segments_per_video[mv] = music_videos.count(mv)
        
        # only save a list of unique music video ids as attribute(without duplicates)
        self.music_videos = np.unique(music_videos)
        # if n is specified, only sample from n videos
        if n:
            self.music_videos = self.music_videos[:n]
        
        # a dictionary that maps an index to a key
        self.idxdict = {name:index for name, index in zip(self.music_videos, range(len(self.music_videos)))}

    def __len__(self):
        """
        return number of audio segments in the dataset (equal to number of video segments)
        """
        return len(self.music_videos)

    def __getitem__(self, 
                    idx: str):
        """
        idx:     xite music video id
        returns: audio and video tensors for a random segment of that video. Also includes an index  
        """
        # select random segment and download audio and video by calling kedro loader
        n_segments = self.segments_per_video[idx]
        #segment_name = f'{idx}_{random.randint(0, n_segments-1)}'
        segment_name = f'{idx}_{0}'
        video_tensor = self.video_loaders[segment_name]()
        audio_tensor = self.audio_loaders[segment_name]()

        return audio_tensor, video_tensor, self.idxdict[idx]

class OneModalEmbeddingDataset(Dataset):
    """
    custom dataset compatible with pytorch for either audio or video embeddings kn the SSD
    loads all embeddings
    """
    def __init__(self, 
                 loaders: Dict,
                 metadata: pandas.DataFrame,
                 n:int=None):
        """
        audio_loaders: kedro lazy data loaders for audio embeddings
        video_loaders: kedro lazy data loaders for video embeddings
        """
        # save function parameters to attributes
        self.loaders = loaders

        # get list of segments
        self.segments = list(self.loaders.keys())

        if n:
            self.segments = self.segments[:n]

        self.idxdict = {name:index for name, index in zip(self.segments, range(len(self.segments)))}
        self.genredict = dict()

        le = preprocessing.LabelEncoder()
        targets = le.fit_transform(metadata['genre_list'])
        # targets: array([0, 1, 2, 3])
        metadata['target'] = targets

        print('finding genre per segment')
        for segment in tqdm(self.segments):
            row = metadata[metadata['videoid'] == int(segment.split('_')[0])]
            genre = row.iloc[0].target

            self.genredict[segment] = genre


    def __len__(self):
        """
        return number of audio segments in the dataset (equal to number of video segments)
        """
        return len(self.segments)

    def __getitem__(self, 
                    idx: str):
        """
        idx:     xite music video id
        returns: audio and video tensors for a random segment of that video. Also includes an index  
        """
        tensor = self.loaders[idx]()

        return tensor, self.idxdict[idx], self.genredict[idx]


class MySampler(Sampler):
    """
    Samples music videos randomly, without replacement. 
    """

    def __init__(self, 
                 dataset: MultiModalDataset,
                 shuffle=True):
        self.music_videos = dataset.music_videos
        random.shuffle(self.music_videos)
        
    def __iter__(self):
        return iter(self.music_videos)

    def __len__(self):
        return len(self.music_videos)


class OneModalSampler(Sampler):
    """
    Samples embeddings (used for downstream tasks)
    """

    def __init__(self, 
                 dataset: OneModalEmbeddingDataset,
                 shuffle=True):
        self.segments = dataset.segments
        random.shuffle(self.segments)
        
    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)


class AllSampler(Sampler):
    """
    Samples segments randomly, without replacement. 
    """

    def __init__(self, 
                 dataset: MultiModalDataset,
                 shuffle=True):
        self.segments = []
        segment_array_2d = dataset.segment_array_2d
        random.shuffle(segment_array_2d)

        # loop though 2d array and select random segments per video
        print('Preventing music video overlap in batches...')
        for i in range(6):
            print(i, end = "\r") 
            for mv_segments in segment_array_2d:
                if len(mv_segments) > 0:
                    # add a random segment to the list of segments and remove it from the 2d array
                    segment_id = random.randrange(len(mv_segments))
                    self.segments.append(mv_segments.pop(segment_id))

    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)


class DownloadDataset(Dataset):
    """
    custom dataset compatible with pytorch for downloading audio and video from bucket (not for training)
    """
    def __init__(self, 
                 audio_transforms: torchvision.transforms.transforms.Compose, 
                 video_transforms: torchvision.transforms.transforms.Compose, 
                 n_audio_samples: int, 
                 n_video_samples: int,
                 audio_loaders: Dict[str, callable],
                 video_loaders: Dict[str, callable],
                 n:int=None,
                 exclude=None):
        """
        bucket_name: name of google cloud storage bucket to load data from
        audio_transforms: augmentations to perform on the audio track
        video_transforms: augmentations to perform on the video
        n_audio_samples: number of audio segments to use per segment
        n_video_samples: number of video segments to use per segment
        audio_loaders: kedro lazy data loaders for audio files
        video_loaders: kedro lazy data loaders for video files
        """
        # save function parameters to attributes
        self.audio_transforms = audio_transforms
        self.video_transforms = video_transforms
        self.n_video_samples = n_video_samples
        self.n_audio_samples = n_audio_samples
        self.clipaudio = ClipAudio(n_audio_samples)
        self.clipvideo = ClipVideo(n_video_samples)
        self.audio_loaders = audio_loaders
        self.video_loaders = video_loaders
        
        # check that all segments have both an mp4 and a wav file
        if len(self.audio_loaders) != len(self.video_loaders):
            warnings.warn("Some segments don't have both an audio and a video file") 
            
        # get list of segments
        self.segments = list(self.audio_loaders.keys())
        if exclude:
            self.segments = [segment for segment in self.segments if segment not in exclude]

        music_videos = np.unique([name.split('_')[0] for name in self.segments])
        # if n is specified, only sample from n videos
        #if n:
        #    music_videos = music_videos[:n]
        #    self.segments = [s for s in self.segments if s.split('_')[0] in music_videos]
        
        # a dictionary that maps an index to a key
        self.idxdict = {name:index for name, index in zip(self.segments, range(len(self.segments)))}

    def __len__(self):
        """
        return number of audio segments in the dataset (equal to number of video segments)
        """
        return len(self.segments)

    def __getitem__(self, 
                    idx: str):
        """
        idx:     xite music video id
        returns: audio and video tensors for a random segment of that video. Also includes an index  
        """
        # select random segment and download audio and video by calling kedro loader
        
        video_tensor = self.video_loaders[idx]()
        audio_tensor, _ = self.audio_loaders[idx]()
        
        # TODO: do this in ffmpeg not when loading data
        audio_tensor = torch.mean(audio_tensor, axis=0)

        # get the latest possible start of the segment (as 0<float<1), and sample a starting point
        latest_possible_start = 1 - (self.n_video_samples/video_tensor.shape[1])
        segment_start = np.random.uniform(0, latest_possible_start)
        
        # cut out synchronous parts from audio and video
        audio_tensor = self.clipaudio(audio_tensor, segment_start)
        video_tensor = self.clipvideo(video_tensor, segment_start)
        
        # apply augmentations to video and audio
        video_tensor = self.video_transforms(video_tensor)
        audio_tensor = self.audio_transforms(audio_tensor[None, None, :])[0, 0, :]

        return audio_tensor, video_tensor, self.idxdict[idx], idx
    
    
class DownloadSampler(Sampler):
    """
    Samples elements randomly, without replacement. 
    Arguments:
        bucket_name: google bucket to sample from
    """

    def __init__(self, 
                 dataset: MultiModalDataset):
        self.segments = dataset.segments

    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)    
 
    
class ClipAudio:
    """
    custom audio augmentation that clips segment from audio
    arguments:   sample rate: sample rate to use
                 clip length: length of the clip to cut   
    """
    def __init__(self, 
                 clip_length: int):
        # save attributes
        self.clip_length = clip_length

    def __call__(self, 
                 audio_data: torch.Tensor, 
                 start: float):
        """
        function that gets called when the transformation is being executed
        arguments:  audio_data: pytorch tensor for the audio segment to be clipped
                    start: float between 0 and 1 indicating where to make the cut
        """
        audio_length = audio_data.shape[0]

        if audio_length > self.clip_length:
            offset = int(start * audio_length)
            if offset > audio_length - self.clip_length:
                offset = audio_length - self.clip_length

            audio_data = audio_data[offset:(offset+self.clip_length)]

        return audio_data
    
    
class ClipVideo:
    """
    custom audio augmentation that clips random segment from audio
    arguments:   sample rate: sample rate to use
                 clip length: length of the clip to cut   
    """
    def __init__(self, 
                 clip_length: int):
        self.clip_length = clip_length

    def __call__(self, 
                 video_data: torch.Tensor, 
                 start: float):
        """
        function that gets called when the transformation is being executed
        arguments:  audio_data: pytorch tensor for the audio segment to be clipped
                    start: float between [0 and clip_length/original_length] indicating where to make the cut
        """
        video_length = video_data.shape[1]

        if video_length > self.clip_length:
            offset = int(start * video_length)
            # TODO: see how this cna be done better, this way the end of the clip is more likely to be shown
            if offset > video_length - self.clip_length:
                offset = video_length - self.clip_length

            video_data = video_data[:, offset:(offset+self.clip_length), :, :]

        return video_data
    
    
class AddGaussianNoise(object):
    """
    augmentation class that adds gaussian noise to video
    """
    def __init__(self, 
                 mean: float=0., 
                 std: float=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, 
                 tensor: torch.Tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    
def create_dataloader(batch_size: int, 
                      audio_loaders, 
                      video_loaders,
                      n: int,
                      opt: Dict):
    """
    node function that creates an audio-visual dataloader

    arguments:  batch_size: size of batches used during optimization
                audio_loaders: kedro loaders for the audio files or alternatively a string 
                video_laoders: kedro loaders for the video files or alternatively a string 
    """
    if opt['use_embeddings']:
        # create dataset and dataloader
        dataset = EmbeddingDataset(audio_loaders=audio_loaders,
                                    video_loaders=video_loaders,
                                    n=n)
    else: 
        # define data augmentations for audio
        atf = transforms.Compose([
            RandomApply([ta.noise.Noise(min_snr=0.001, max_snr=0.03)], p=0.3),
            RandomApply([ta.gain.Gain(min_gain=-20, max_gain=-1)]),
            RandomApply([ta.pitch_shift.PitchShift(n_samples=16000*3,
                                                sample_rate=16000, 
                                                pitch_shift_min= -7.0,
                                                pitch_shift_max = 7.0,
                                                bins_per_octave = 12)], p=.3)
        ])


        # define augmentations for video
        vtf = transforms.Compose([
            RandomResizedCrop(target_height=112, 
                            target_width=112,
                            scale=(.5,.5),
                            aspect_ratio=(1,1)),
            transforms.RandomHorizontalFlip(.5),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma =(1,2)),
            AddGaussianNoise(0,6),
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))

        ])
        # create dataset and dataloader
        dataset = MultiModalDataset(audio_transforms=atf, 
                                    video_transforms=vtf,
                                    n_audio_samples=16000*3,
                                    n_video_samples=30,
                                    audio_loaders=audio_loaders,
                                    video_loaders=video_loaders,
                                    n=n)
 
    sampler = MySampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size, True)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    return dataloader
