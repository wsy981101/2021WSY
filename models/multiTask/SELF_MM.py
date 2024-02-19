# self supervised multimodal multi-task learning network
import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.BertTextEncoder import BertTextEncoder

__all__ = ['SELF_MM', 'TA_GML']

    
class TA_GML(nn.Module):
    def __init__(self, args):
        super(TA_GML, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)
        self.ta = args.ta
        self.cross = not args.cross

        # audio-vision subnets
        audio_in, video_in = args.feature_dims[1:]
        if self.ta:
            self.audio_model = AuViTASubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
                                num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
            self.video_model = AuViTASubNet(video_in, args.v_lstm_hidden_size, args.video_out, \
                                num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)
        else:
            self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
            self.video_model = AuViSubNet(video_in, args.v_lstm_hidden_size, args.video_out, \
                                num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        if self.cross:
            self.post_fusion_layer_1 = nn.Linear((args.text_out + args.video_out + args.audio_out)*3, args.post_fusion_dim)
        else:
            self.post_fusion_layer_1 = nn.Linear(args.text_out + args.video_out + args.audio_out, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(args.text_out, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.audio_out, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(args.video_out, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim, 1)

        # the generator for fake label
        self.expert_fusion = nn.Linear(args.post_fusion_dim, args.expert_dim)
        self.expert_text = nn.Linear(args.post_text_dim, args.expert_dim)
        self.expert_audio = nn.Linear(args.post_audio_dim, args.expert_dim)
        self.expert_video = nn.Linear(args.post_video_dim, args.expert_dim)

        if self.ta:
            self.alpha1 = nn.Sequential(
                    nn.Linear(args.text_out, 1, bias=True),
                    nn.Tanh()
                )
            self.beta1 = nn.Sequential(
                    nn.Linear(args.text_out, 1, bias=True),
                    nn.Tanhshrink()
                )

    def forward(self, text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video

        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach().cpu()

        if self.ta:
            text = self.text_model(text)
            alpha_output = self.alpha1(text)
            beta_output = self.beta1(text)
            text = text*alpha_output*beta_output
            text = text[:,0,:]
        else:
            text = self.text_model(text)[:,0,:]
        

        if self.aligned:
            audio = self.audio_model(audio, text_lengths)
            video = self.video_model(video, text_lengths)
        else:
            audio = self.audio_model(audio, audio_lengths)
            video = self.video_model(video, video_lengths)
        # fusion
        if self.cross:
            # generate cross feature for T,A,V [Batch, dim]
            TA = (audio.unsqueeze(1) @ torch.tile(text, (1, audio.shape[1])).view(-1, audio.shape[1], text.shape[1])).squeeze()
            AT = (text.unsqueeze(1) @ torch.tile(audio, (1, text.shape[1])).view(-1, text.shape[1], audio.shape[1])).squeeze()
            TV = (video.unsqueeze(1) @ torch.tile(text, (1, video.shape[1])).view(-1, video.shape[1], text.shape[1])).squeeze()
            VT = (text.unsqueeze(1) @ torch.tile(video, (1, text.shape[1])).view(-1, text.shape[1], video.shape[1])).squeeze()
            AV = (video.unsqueeze(1) @ torch.tile(audio, (1, video.shape[1])).view(-1, video.shape[1], audio.shape[1])).squeeze()
            VA = (audio.unsqueeze(1) @ torch.tile(video, (1, audio.shape[1])).view(-1, audio.shape[1], video.shape[1])).squeeze()

            fusion_h = torch.cat([text, audio, video, TA, AT, TV, VT, AV, VA], dim=-1)
            
        else:
            fusion_h = torch.cat([text, audio, video], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)#jian shao guo ni he feng xian
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # # text
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)

        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        # the expert feature for fake label
        e_f = F.relu(self.expert_fusion(fusion_h), inplace=False)
        e_t = F.relu(self.expert_text(text_h), inplace=False)
        e_a = F.relu(self.expert_audio(audio_h), inplace=False)
        e_v = F.relu(self.expert_video(video_h), inplace=False)

        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
            'Expert_f': e_f,
            'Expert_t': e_t,
            'Expert_a': e_a,
            'Expert_v': e_v,
        }
        return res

class SELF_MM(nn.Module):
    def __init__(self, args):
        super(SELF_MM, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        # audio-vision subnets
        audio_in, video_in = args.feature_dims[1:]
        self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_model = AuViSubNet(video_in, args.v_lstm_hidden_size, args.video_out, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(args.text_out + args.video_out + args.audio_out, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(args.text_out, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.audio_out, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(args.video_out, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim, 1)

    def forward(self, text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video

        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach().cpu()
        text = self.text_model(text)[:,0,:]

        if self.aligned:
            audio = self.audio_model(audio, text_lengths)
            video = self.video_model(video, text_lengths)
        else:
            audio = self.audio_model(audio, audio_lengths)
            video = self.video_model(video, video_lengths)
        
        # fusion
        fusion_h = torch.cat([text, audio, video], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # # text
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)

        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
        }
        return res

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x.cpu(), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence.to(x.device))
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1

class AuViTASubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViTASubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

        self.alpha1 = nn.Sequential(
                nn.Linear(hidden_size, 1, bias=True),
                nn.Tanh()
            )
        self.beta1 = nn.Sequential(
                nn.Linear(hidden_size, 1, bias=True),
                nn.Tanhshrink()
            )

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x.cpu(), lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, final_states = self.rnn(packed_sequence.to(x.device))
        output = pad_packed_sequence(output, batch_first=False)[0]
        alpha_output = self.alpha1(output.data)
        beta_output = self.beta1(output.data)
        output = output.data*alpha_output*beta_output
        h = self.dropout(output[-1].squeeze())
        y_1 = self.linear_1(h)
        return y_1
