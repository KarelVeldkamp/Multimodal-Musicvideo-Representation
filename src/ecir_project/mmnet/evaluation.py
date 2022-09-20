from __future__ import print_function
import numpy
import time
import numpy as np
import torch
from collections import OrderedDict
import os
from thesis_project.mmnet.model import MMNet


class AverageMeter():
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, 
               val: float, 
               n: int=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector():
    """
    A collection of logging objects that can change from train to val
    """

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """
        Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s



def encode_data(model: MMNet, 
                data_loader: torch.utils.data.DataLoader, 
                log_step: int=10):
    """
    Encode all videos and audio tracks loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    vid_embs = None
    aud_embs = None
    for i, (audios, videos, idx) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        with torch.no_grad():
            vid_emb, aud_emb = model.forward_emb(videos, audios)

        # initialize the numpy arrays given the size of the embeddings
        if vid_embs is None:
            vid_embs = np.zeros((len(data_loader.dataset), vid_emb.size(1)))
            aud_embs = np.zeros((len(data_loader.dataset), aud_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        vid_embs[idx] = vid_emb.data.cpu().numpy().copy()
        aud_embs[idx] = aud_emb.data.cpu().numpy().copy()

        # measure accuracy and record loss
        model.forward_loss(vid_emb, aud_emb)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #if i % log_step == 0:
        #   wandb.log({
        #       'Batch': i,
        #       'BatchTime': batch_time.val,
        #   })
        del videos, audios

    return vid_embs, aud_embs

def get_validation_metrics(videos: torch.Tensor, 
                           audios: torch.Tensor,
                           batch_size: int):
    """
    My alternative for a2v and v2a. Calculates the cosine simlarity between all video audio pairs, and uses the similarity 
    matrix to calculate median rank, mean rank, recall@1, recall@5 and recall@10 for both audio retrieval and video retrieval. 
    
    arguments:    videos: (N, embedding size) Matrix with video embeddings
                  audios: (N, embedding size) Matrix with audio embeddings
                  
    returns:      dictionary of ranking metrics.
    """

    niter = videos.shape[0] // batch_size 

    # set up meters to track average over batches
    median_rank_v2a = AverageMeter()
    q25_rank_v2a = AverageMeter()
    q75_rank_v2a = AverageMeter()
    mean_rank_v2a = AverageMeter()
    r1_v2a = AverageMeter()
    r5_v2a = AverageMeter()
    r10_v2a = AverageMeter()
    median_rank_a2v = AverageMeter()
    mean_rank_a2v = AverageMeter()
    r1_a2v = AverageMeter()
    r5_a2v = AverageMeter()
    r10_a2v = AverageMeter()
    q25_rank_a2v = AverageMeter()
    q75_rank_a2v = AverageMeter()

    for i in range(niter):
        first = batch_size * i
        last = batch_size * (i+1)
        video_batch = videos[first:last, :]
        audio_batch = audios[first:last, :]

        # calculate cosine similarity between all video-audio pairs
        cos_sim = numpy.dot(video_batch, audio_batch.T)
        
        ### First I calculate metrics for video->audio retrieval:
        # get a vector with the rank of the matching audio segment for each query video segment
        ranks_per_row = np.argsort(cos_sim, axis=1)
        v2a_ranks = np.diag(ranks_per_row)
    
        # calculate median and mean rank
        median_rank_v2a.update(np.median(v2a_ranks))
        mean_rank_v2a.update(np.mean(v2a_ranks))

        # calculate quantiles
        q25_rank_v2a.update(np.quantile(v2a_ranks, .25))
        q75_rank_v2a.update(np.quantile(v2a_ranks, .75))

        # calculate recall@1, recall@5 and recall@10
        r1_v2a.update(np.mean(v2a_ranks==0))
        r5_v2a.update(np.mean(v2a_ranks<5))
        r10_v2a.update(np.mean(v2a_ranks<10))
        
        ### Now I do the save for audio -> video retrieval
        # get a vector with the rank of the matching video segment for each query audio segment
        ranks_per_col = np.argsort(cos_sim, axis=0)
        a2v_ranks = np.diag(ranks_per_row)
        
        # calculate median and mean rank
        median_rank_a2v.update(np.median(a2v_ranks))
        mean_rank_a2v.update(np.mean(a2v_ranks))

        # calculate quantiles
        q25_rank_a2v.update(np.quantile(a2v_ranks, .25))
        q75_rank_a2v.update(np.quantile(a2v_ranks, .75))

        # calculate recall@1, recall@5 and recall@10
        r1_a2v.update(np.mean(a2v_ranks==0))
        r5_a2v.update(np.mean(a2v_ranks<5))
        r10_a2v.update(np.mean(a2v_ranks<10))

    
    return dict(
        median_rank_v2a=median_rank_v2a.avg,
        q25_v2a=q25_rank_v2a.avg,
        q75_v2a=q75_rank_v2a.avg,
        mean_rank_v2a=mean_rank_v2a.avg,
        r1_v2a=r1_v2a.avg,
        r5_v2a=r5_v2a.avg,
        r10_v2a=r10_v2a.avg,
        median_rank_a2v=median_rank_a2v.avg,
        q25_a2v=q25_rank_a2v.avg,
        q75_a2v=q75_rank_a2v.avg,
        mean_rank_a2v=mean_rank_a2v.avg,
        r1_a2v=r1_a2v.avg,
        r5_a2v=r5_a2v.avg,
        r10_a2v=r10_a2v.avg
    )
