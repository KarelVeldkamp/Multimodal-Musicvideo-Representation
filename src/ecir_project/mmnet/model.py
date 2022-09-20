import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init
import torchvision
import torchaudio
import torch.backends.cudnn as cudnn

def l2norm(X: torch.Tensor):
    """
    L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def cosine_sim(videos: torch.Tensor, 
               audios: torch.Tensor):
    """
    Cosine similarity between all the video audio pairs (Just dot product since they are normalised). 
    """
    return videos.mm(audios.t())


class AudioEncoder(nn.Module):
    """
    Class for the audio modality encoder. Consists of musicnn convolutional layers with three dense layers on top.
    """
    
    def __init__(self, 
                 layer_sizes: list=[512, 256], 
                 normalise: bool=True,
                 sample_rate: int=16000,
                 n_fft: int=512,
                 f_min: float=0.0,
                 f_max: float=8000.0, 
                 n_mels: int=96,
                 dropout: float=.5):
        """
        layer_sizes: layer sizes for the three dense layers, the last one being the embedding size
        normalise: whether to l2 normalise the output vectors
        sample_rate: sample rate of the audio track
        n_fft: number of frequency bins for the fft
        f_min: minimum frequency
        f_max: maximum frequency
        n_mels: number of filterbanks
        dropout: dropout rate for the dense layers of the network
        
        """
        
        super(AudioEncoder, self).__init__()
        
        self.dense1_size, self.embed_size = layer_sizes
        self.normalise = normalise
        
        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # Pons musicnn front-end
        m1 = Conv_V(1, 204, (int(0.7*96), 7))
        m2 = Conv_V(1, 204, (int(0.4*96), 7))
        m3 = Conv_H(1, 51, 129)
        m4 = Conv_H(1, 51, 65)
        m5 = Conv_H(1, 51, 33)
        self.layers = nn.ModuleList([m1, m2, m3, m4, m5])

        # Pons musicnn back-end
        self.layer1 = Conv_1d(561, 512, 7, 1, 1)
        self.layer2 = Conv_1d(512,512, 7, 1, 1)
        self.layer3 = Conv_1d(512, 512, 7, 1, 1)
        
        # Additional dense connected layers
        self.dense1 = nn.Linear((561+(512*3))*2, self.dense1_size)
        self.bn1 = nn.BatchNorm1d(self.dense1_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(self.dense1_size, self.embed_size)

        # specify which layers form the projection head
        self.proj_head_params = []
        self.proj_head_params += list(self.dense1.parameters())
        self.proj_head_params += list(self.dense2.parameters())

    
    
    def forward(self, 
                x: torch.Tensor):
        
        # spectogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # Pons front-end
        out = []
        for layer in self.layers:
            out.append(layer(x))
        out = torch.cat(out, dim=1)

        # Pons back-end
        length = out.size(2)
        res1 = self.layer1(out)
        res2 = self.layer2(res1) + res1
        res3 = self.layer3(res2) + res2
        out = torch.cat([out, res1, res2, res3], 1)

        mp = nn.MaxPool1d(length)(out)
        avgp = nn.AvgPool1d(length)(out)

        out = torch.cat([mp, avgp], dim=1)
        out = out.squeeze(2)

        # dense layers
        out = self.relu(self.bn1(self.dense1(out)))
        out = self.dropout(out)
        out = self.dense2(out)
        out = nn.Sigmoid()(out)
        
        if self.normalise:
            out = l2norm(out)
        
        return out
        
        

class VideoEncoder(nn.Module):
    """
    Class for the video modality encoder. Consists of musicnn convolutional layers with three dense layers on top.
    """
    
    def __init__(self, 
                 layer_sizes: list, 
                 normalise: bool=True,
                 dropout: float=.3):
        """
        layer_sizes: list of dense layer sizes, last one being the embedding size
        normalise: whether to l2 normalise the output od the network
        """
        
        super(VideoEncoder, self).__init__()
        # set attibutes
        self.normalise = normalise
        self.dense1_size, self.embed_size = layer_sizes
        self.dropout = nn.Dropout(dropout)

        # get (2+1)D cnn architecture
        self.r2plus1 = torchvision.models.video.r2plus1d_18()
        
        # Add three dense connected layers with match normalisation and relu
        self.r2plus1.fc = nn.Linear(512, self.dense1_size)
        self.bn1 = nn.BatchNorm1d(self.dense1_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(self.dense1_size, self.embed_size)

        # specify which layers form the projection head
        self.proj_head_params = []
        self.proj_head_params += list(self.r2plus1.fc.parameters())
        self.proj_head_params += list(self.dense2.parameters())
        
    
    def forward(self, 
                x: torch.Tensor):
        # pretrained network plus classification layers
        out = self.relu(self.bn1(self.r2plus1(x)))
        out = self.dropout(out)
        out = self.dense2(out)
        out = nn.Sigmoid()(out)
        
        if self.normalise:
            out =  l2norm(out)
        
        return out


class AudioEncoderEmb(nn.Module):
    """
    Class for the audio modality encoder that uses precomputed embeddings. 
    """
    
    def __init__(self, 
                 layer_sizes: list=[512, 256], 
                 normalise: bool=True,
                 dropout: float=.3):
        """
        layer_sizes: layer sizes for the three dense layers, the last one being the embedding size
        normalise: whether to l2 normalise the output vectors
        dropout: dropout rate for the dense layers of the network
        """
        
        super(AudioEncoderEmb, self).__init__()
        
        self.dense1_size, self.embed_size = layer_sizes
        self.normalise = normalise
        self.dropout = nn.Dropout(dropout)
        
        # Add three dense connected layers with match normalisation and relu
        self.dense1 = nn.Linear((561+(64*3))*2, self.dense1_size)
        self.bn1 = nn.BatchNorm1d(self.dense1_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(self.dense1_size, self.embed_size)

        # specify which layers form the projection head
        self.proj_head_params = []
        self.proj_head_params += list(self.dense1.parameters())
        self.proj_head_params += list(self.dense2.parameters())

   
    def forward(self, 
                x: torch.Tensor):

        # dense layers
        out = self.relu(self.bn1(self.dense1(x)))
        out = self.dropout(out)
        out = self.dense2(out)
        out = nn.Sigmoid()(out)
        
        if self.normalise:
            out = l2norm(out)
        
        return out
        

class VideoEncoderEmb(nn.Module):
    """
    Class for the video modality encoder based on precomputed embeddings. 
    """
    
    def __init__(self, 
                 layer_sizes: list, 
                 normalise: bool=True,
                 dropout: float=.3):
        """
        layer_sizes: list of dense layer sizes, last one being the embedding size
        normalise: whether to l2 normalise the output od the network
        """
        
        super(VideoEncoderEmb, self).__init__()
        # set attibutes
        self.normalise = normalise
        self.dense1_size, self.embed_size = layer_sizes
        self.dropout = nn.Dropout(dropout)
        
        # Add three dense connected layers with match normalisation and relu
        self.dense1 = nn.Linear(512, self.dense1_size)
        self.bn1 = nn.BatchNorm1d(self.dense1_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(self.dense1_size, self.embed_size)
        
        # specify which layers form the projection head
        self.proj_head_params = []
        self.proj_head_params += list(self.dense1.parameters())
        self.proj_head_params += list(self.dense2.parameters())

        
    
    def forward(self, 
                x: torch.Tensor):

        # classification layers
        out = self.relu(self.bn1(self.dense1(x)))
        out = self.dropout(out)
        out = self.dense2(out)
        out = nn.Sigmoid()(out)
        
        if self.normalise:
            out =  l2norm(out)
        
        return out
        
    
class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, sim_metric: callable=cosine_sim,
                 max_violation: bool=False, 
                 temperature: float=1.0):
        
        super(ContrastiveLoss, self).__init__()
        
        self.sim = sim_metric
        self.max_violation = max_violation
        self.temperature = temperature
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, 
                video: torch.Tensor, 
                audio: torch.Tensor):

        # compute cosine similarities
        logits = torch.matmul(audio, video.t()) / self.temperature
        # get correct labels
        labels = torch.arange(video.shape[0])
        if torch.cuda.is_available():
            labels = labels.cuda()
        
        # calculate cross entropy loss both from audio to video and video to audio
        loss_a = self.loss(logits, labels)
        loss_v = self.loss(logits.t(), labels)
        
        return loss_a + loss_v


class AudioEncoderEmb2(nn.Module):
    """
    Class for the audio modality encoder that uses precomputed embeddings. 
    """
    
    def __init__(self):
        """
        layer_sizes: layer sizes for the three dense layers, the last one being the embedding size
        normalise: whether to l2 normalise the output vectors
        dropout: dropout rate for the dense layers of the network
        """
        
        super(AudioEncoderEmb2, self).__init__()
        
        self.layer = nn.Identity()

        # specify which layers form the projection head
        self.proj_head_params = []
        self.proj_head_params += list(self.layer.parameters())


    def forward(self, 
                x: torch.Tensor):
        return x


class VideoEncoderEmb2(nn.Module):
    """
    Class for the video modality encoder based on precomputed embeddings. 
    """
    
    def __init__(self, 
                 layer_sizes: list, 
                 normalise: bool=True,
                 dropout: float=.3):
        """
        layer_sizes: list of dense layer sizes, last one being the embedding size
        normalise: whether to l2 normalise the output od the network
        """
        
        super(VideoEncoderEmb2, self).__init__()
        # set attibutes
        self.normalise = normalise
        self.dense1_size, self.dense2_size, self.embed_size = layer_sizes
        self.dropout = nn.Dropout(dropout)
        
        # Add three dense connected layers with match normalisation and relu
        self.dense1 = nn.Linear(512, self.dense1_size)
        self.bn1 = nn.BatchNorm1d(self.dense1_size)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(self.dense1_size, self.dense2_size)
        self.dense3 = nn.Linear(self.dense2_size, self.embed_size)
        
        # specify which layers form the projection head
        self.proj_head_params = []
        self.proj_head_params += list(self.dense1.parameters())
        self.proj_head_params += list(self.dense2.parameters())
        self.proj_head_params += list(self.dense3.parameters())
        
    
    def forward(self, 
                x: torch.Tensor):

        # classification layers
        out = self.relu(self.bn1(self.dense1(x)))
        out = self.dropout(out)
        out = self.dense2(out)
        out = nn.Sigmoid()(out)
        
        if self.normalise:
            out =  l2norm(out)
        
        return out
        
    
class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, sim_metric: callable=cosine_sim,
                 max_violation: bool=False, 
                 temperature: float=1.0):
        
        super(ContrastiveLoss, self).__init__()
        
        self.sim = sim_metric
        self.max_violation = max_violation
        self.temperature = temperature
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, 
                video: torch.Tensor, 
                audio: torch.Tensor):

        # compute cosine similarities
        logits = torch.matmul(audio, video.t()) / self.temperature
        # get correct labels
        labels = torch.arange(video.shape[0])
        if torch.cuda.is_available():
            labels = labels.cuda()
        
        # calculate cross entropy loss both from audio to video and video to audio
        loss_a = self.loss(logits, labels)
        loss_v = self.loss(logits.t(), labels)
        
        return loss_a + loss_v


class MMNet(object):
    """
    The multimodal network
    """

    def __init__(self, 
                 opt: dict,
                 video_encoder,
                 audio_encoder):
        """
        opt: dictionary of optimization parmaters, containing keys:
            video_layer_sizes: list of len 3 for dense video encoder layers, last one is embedding size
            video_layer_sizes: same for audio layers, embedding size should be equal over modalities
            normalise: whether to normalise the output vectors of the encoders
            finetune: whether to finetune all layers (alternatively just teh dense layers are trained)
            learning_rate: the learning rate
            
        """
        # Build Models 
        self.vid_enc = video_encoder
        self.aud_enc = audio_encoder
        
        # move to gpu 
        if torch.cuda.is_available(): 
            self.vid_enc.cuda()
            self.aud_enc.cuda()
            cudnn.benchmark = True

        # Loss 
        self.criterion = ContrastiveLoss()
        
        # determine which paramters to train, and add those to optimizer 
        params = []
        if opt['finetune']:
            params += self.aud_enc.parameters()
            params += self.vid_end.parameters()
        else:
            params += self.aud_enc.proj_head_params
            params += self.vid_enc.proj_head_params
           
        self.optimizer = torch.optim.Adam(params, lr=opt['learning_rate'])

        # iterations
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.vid_enc.state_dict(), self.aud_enc.state_dict()]
        return state_dict

    def load_state_dict(self, 
                        state_dict: dict):
        self.vid_enc.load_state_dict(state_dict[0])
        self.aud_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """
        switch to train mode
        """
        self.vid_enc.train()
        self.aud_enc.train()

    def val_start(self):
        """
        switch to evaluate mode
        """
        self.vid_enc.eval()
        self.aud_enc.eval()

    def forward_emb(self, 
                    videos: torch.Tensor, 
                    audios: torch.Tensor,
                    volatile: bool=False):
        """
        Compute the video and audio embeddings
        """
        # Set mini-batch dataset
        videos = Variable(videos, volatile=volatile)
        audios = Variable(audios, volatile=volatile)
        if torch.cuda.is_available():
            videos = videos.cuda()
            audios = audios.cuda()

        # Forward
        vid_emb = self.vid_enc(videos)
        aud_emb = self.aud_enc(audios)
        return vid_emb, aud_emb

    def forward_loss(self, 
                     vid_emb: torch.Tensor, 
                     aud_emb: torch.Tensor, 
                     **kwargs):
        """
        Compute the loss given pairs of video and audio embeddings
        """
        loss = self.criterion(vid_emb, aud_emb)
        #self.logger.update('Le', loss.data[0], vid_emb.size(0))
        return loss

    def train_emb(self, 
                  audios: torch.Tensor, 
                  videos: torch.Tensor,
                  *args):
        """
        One training step given videos and audio.
        """
        self.Eiters += 1
        #self.logger.update('Eit', self.Eiters)
        #self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        vid_emb, aud_emb = self.forward_emb(videos, audios)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(vid_emb, aud_emb)

        # compute gradient and do SGD step
        loss.backward()

        self.optimizer.step()
        
        # return loss for logging
        return loss    


class Conv_1d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_1d, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out
    

class Conv_V(nn.Module):
    """
    Class for vertical convolutional layer, from https://github.dev/fartashf/vsepp/
    """
    # vertical convolution
    def __init__(self, input_channels, output_channels, filter_shape):
        super(Conv_V, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, filter_shape,
                              padding=(0, filter_shape[1] // 2))
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        freq = x.size(2)
        out = nn.MaxPool2d((freq, 1), stride=(freq, 1))(x)
        out = out.squeeze(2)
        return out





class Conv_H(nn.Module):
    """
    Class for horizontal convolutional layer, from https://github.dev/fartashf/vsepp/
    """
    # horizontal convolution
    def __init__(self, input_channels, output_channels, filter_length):
        super(Conv_H, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, filter_length,
                              padding=filter_length // 2)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        freq = x.size(2)
        out = nn.AvgPool2d((freq, 1), stride=(freq, 1))(x)
        out = out.squeeze(2)
        out = self.relu(self.bn(self.conv(out)))
        return out



