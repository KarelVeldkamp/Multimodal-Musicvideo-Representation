"""
This is a boilerplate pipeline 'download_embeddings'
generated using Kedro 0.17.7
"""
from thesis_project.mmnet.model import *
from thesis_project.mmnet.data import *
from thesis_project.sota_models.model import MusicnnNoClassifier
import torchvision
from sqlitedict import SqliteDict
from tqdm import tqdm
from typing import Dict
import sys
import gc
from datetime import datetime


# A neural network layes that does nothing, used to replace pretrained classification layer
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def download_embeddings(audio_loaders: Dict[str, callable], 
                        video_loaders: Dict[str, callable], 
                        n: int, 
                        batch_size: int, 
                        subset: str):
    """
    function that dowloads audio and video segments from the google cloud bucket, computes the embeddings using 
    the pretrained networks, and stores them in an sqlite dict.
    arguments:  audio loaders: dictionary of kedro audio loaders
                video loaders: dictionary of kedro video loaders
                n: number of music videos to download 
                batch_size: batch size
                subset: train, validation or test
    """

    if n == 0:
        return 

    # load audio encoder
    weights = torch.load('data/06_models/musicnn_best_model.pth', map_location=torch.device('cuda'))
    musicnn = MusicnnNoClassifier().cuda()
    musicnn.load_state_dict(weights, strict=False)
    
    # load video encoder
    r2plus1d = torchvision.models.video.r2plus1d_18(pretrained=True).cuda()
    r2plus1d.fc = Identity() 
    
    # create dataset and dataloader
    atf = transforms.Compose([
        RandomApply([ta.noise.Noise(min_snr=0.001, max_snr=0.002)], p=0.00001),
    ])
    vtf = transforms.Compose([
        RandomResizedCrop(target_height=112, 
                          target_width=112,
                          scale=(.5,.5),
                          aspect_ratio=(1,1))
    ])

    audio_segments = [f.split('.')[0] for f in os.listdir(f'/data-gpu/audio_{subset}')]
    video_segments =  [f.split('.')[0] for f in os.listdir(f'/data-gpu/video_{subset}')]

    # check that each segment has both a video and an audio segment
    if audio_segments != video_segments:
      raise Exception("Some segments dont have both an audio and a video segment")

    # create dataset and dataloader
    dataset = DownloadDataset(audio_transforms=atf, 
                                video_transforms=vtf,
                                n_audio_samples=16000*3,
                                n_video_samples=30,
                                audio_loaders=audio_loaders,
                                video_loaders=video_loaders,
                                n=n,
                                exclude=audio_segments)
    # creating d
    sampler = DownloadSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size, True)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    # loop though batches and infer embeddings
    for batch in tqdm(dataloader):
        audio, video, _, segment_id = batch
        
        audio = audio
        video = video
        with torch.no_grad():
            aud_emb = musicnn(audio.cuda()).cpu()
            vid_emb = r2plus1d(video.cuda()).cpu()
        
        for i in range(len(segment_id)):
            torch.save(aud_emb[i], f'/data-gpu/audio_{subset}/{segment_id[i]}.pt')
            torch.save(vid_emb[i], f'/data-gpu/video_{subset}/{segment_id[i]}.pt')

        gc.collect()

    
    
