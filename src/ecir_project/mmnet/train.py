from thesis_project.mmnet.data import *
from thesis_project.mmnet.model import *
from thesis_project.mmnet.evaluation import *
from pytorchvideo.transforms import RandomResizedCrop, UniformTemporalSubsample
from torchvision import transforms
from torch.utils.data.sampler import Sampler, BatchSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import Identity
import logging
import shutil
import time
import wandb


def train(opt: dict, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader,
          val_batch_size):
    """
    function that trains the multimodal network

    arguments:  opt: optimization parameters
                train_dataloader: dataloader for the training set
                model: the model instance
                val_dataloader: dataloader for the validation set
    """
    
    # Construct the model
    video_encoder = VideoEncoderEmb2(opt['video_layer_sizes'], opt['normalise'])
    audio_encoder = AudioEncoderEmb2()

    model = MMNet(opt, video_encoder, audio_encoder)
    scheduler = CosineAnnealingLR(model.optimizer, opt['num_epochs'], eta_min=0.005)

    # keep track of the best loss, last epochs loss and of how many times  the loss has increased in a row
    loss_v = 0
    best_loss_v = 1e9
    loss_counter = 0
    # Train the Model
    for epoch in range(opt['num_epochs']):
        
        model.optimizer.step()

        # train for one epoch
        train_epoch(opt, train_dataloader, model, epoch, val_dataloader)

        # evaluate on validation set
        loss_v = validate(opt, train_dataloader, val_dataloader, model, val_batch_size)

        # remember best R@ sum and save checkpoint
        is_best = loss_v < best_loss_v
        best_loss_v = min(loss_v, best_loss_v)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_loss_v,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename=opt['filename'], prefix=opt['logger_name'] + '/')
        
        # increase the counter if loss increased, and possible stop
        if loss_v > best_loss_v:
            loss_counter += 1
            if loss_counter >= opt['max_increases']:
                break
        else:
            loss_counter = 0
            best_loss_v = loss_v

    return model


def train_epoch(opt: dict, 
                train_loader: torch.utils.data.DataLoader, 
                model: MMNet, 
                epoch: int, 
                val_loader: torch.utils.data.DataLoader):
    """
    function that trains for a single epoch
    arguments:  opt: dictionary of optimization parameters, containing keys:
                    reset_train: whether to reset to training mode for each batch
                    log_step: every how many steps to print the logging data   
                train_laoder: pytorch dataloader for training data
                model: multimodal model instance
                epoch: current epoch number
                val_loader: pytorch dataloader for validation data
    """
    
    run = wandb.init(project="AudioVisual Network ", entity="karelveldkamp")
    # average meters to record the training statistics
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    end = time.time()
    # loopt though training data
    for i, train_data in enumerate(train_loader):
        if opt['reset_train']:
            # Always reset to train mode, this is not the default behavior
            model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        loss = model.train_emb(*train_data)
        loss_meter.update(loss)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        # Print log info
        if model.Eiters % opt['log_step'] == 0:

            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))
        # send log to weights and biases
        wandb.log({
            'BatchTime': batch_time.val,
            'DataTime': data_time.val
        })
    # log average loss at the end of epoch
    wandb.log({
        'Train loss': loss_meter.avg,
        'Epoch': epoch
    })
        

def validate(opt: dict, 
             train_loader: torch.utils.data.DataLoader, 
             val_loader: torch.utils.data.DataLoader,  
             model: MMNet,
             val_batch_size):
    """
    calculate recall@1, 5 and 10 for both audio and video retrieval. On the training and validation data
    arguments:  opt: dictionary of optimisation paramters containing keys:
                    log_step: how often to log data
                    measure: the distance measure used for calculating recall
    """
    model.val_start()
    # compute the encoding for all the validation videos and audio tracks
    vid_embs_train, aud_embs_train = encode_data(
        model, train_loader, opt['log_step'])
    
    vid_embs_val, aud_embs_val = encode_data(
        model, val_loader, opt['log_step'])
    model.train_start()


    # calculate validation loss:
    niter = vid_embs_val.shape[0] // val_batch_size 
    val_loss = AverageMeter()
    for i in range(niter):
        first = val_batch_size*i 
        last = val_batch_size*(i+1)
        with torch.no_grad():
            loss = model.forward_loss(torch.Tensor(aud_embs_val[first:last, :]).cuda(), torch.Tensor(vid_embs_val[first:last, :]).cuda())
            val_loss.update(loss)
   
    train_metrics = get_validation_metrics(vid_embs_train, aud_embs_train, val_batch_size)
    validation_metrics = get_validation_metrics(vid_embs_val, aud_embs_val, val_batch_size)
    
    # log metrics to weights and biases
    wandb.log({
        'v2a Train Recall@1': train_metrics['r1_v2a'],
        'v2a Train Recall@5': train_metrics['r5_v2a'],
        'v2a Train Recall@10': train_metrics['r10_v2a'],
        'v2a Train median rank': train_metrics['median_rank_v2a'],
        'v2a Train q25': train_metrics['q25_v2a'],
        'v2a Train q75': train_metrics['q75_v2a'],
        'v2a Train mean rank': train_metrics['mean_rank_v2a'],
        'a2v Train Recall@1': train_metrics['r1_a2v'],
        'a2v Train Recall@5': train_metrics['r5_a2v'],
        'a2v Train Recall@10': train_metrics['r10_a2v'],
        'a2v Train median rank': train_metrics['median_rank_a2v'],
        'a2v Train q25': train_metrics['q25_a2v'],
        'a2v Train q75': train_metrics['q75_a2v'],
        'a2v Train mean rank': train_metrics['mean_rank_a2v'],
        'v2a Validation Recall@1': validation_metrics['r1_v2a'],
        'v2a Validation Recall@5': validation_metrics['r5_v2a'],
        'v2a Validation Recall@10': validation_metrics['r10_v2a'],
        'v2a Validation median rank': validation_metrics['median_rank_v2a'],
        'v2a Validation q25': validation_metrics['q25_v2a'],
        'v2a Valodation q75': validation_metrics['q75_v2a'],
        'v2a Validation mean rank': validation_metrics['median_rank_v2a'],
        'a2v Validation Recall@1': validation_metrics['r1_a2v'],
        'a2v Validation Recall@5': validation_metrics['r5_a2v'],
        'a2v Validation Recall@10': validation_metrics['r10_a2v'],
        'a2v Validation median rank': validation_metrics['median_rank_a2v'],
        'a2v Validation q25': validation_metrics['q25_a2v'],
        'a2v Validation q75': validation_metrics['q75_a2v'],
        'a2v: Validation mean rank': validation_metrics['median_rank_a2v'],
        'Validation loss': val_loss.avg
    })

    return val_loss.avg
    


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
