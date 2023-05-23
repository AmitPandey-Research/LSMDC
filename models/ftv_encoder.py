import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
#import ipdb
import copy


class FTV_encoder(nn.Module):


    def positionalencoding1d_hf(self, d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe

    def positionalencoding1d_lf(self,d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(2000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe


    def __init__(self, config):
        super().__init__()
        # load config parameters
        self.nvid = config.nvid # 5
        self.model_dim = config.encoding_size 
        self.model_type = config.model_type
        self.nhead = 8 # 4
        self.hidden_layers_dim = 2048
        self.dropout_p = 0.1 # 0.2 
        self.nlayers = 6 # 2
        self.batch_size = config.batch_size
        self.nsegments = config.nsegments # 5

        # Token Embedding - Special tokens, video_id, segment_id embeddings
        self.spcl_embed = nn.Embedding(self.nvid, self.model_dim)
        self.video_embed = nn.Embedding(self.nvid, self.model_dim)#.requires_grad_(False)
        self.segment_embed = nn.Embedding(self.nsegments, self.model_dim)#.requires_grad_(False)

        self.video_embed_values = self.positionalencoding1d_lf(self.model_dim,self.nvid)
        self.frame_embed_values = self.positionalencoding1d_hf(self.model_dim,self.nvid)
        
        #self.segment_embed.weight.data = self.frame_embed_values
        #self.video_embed.weight.data = self.video_embed_values

        #self.encoder_layers = TransformerEncoderLayer(self.model_dim, self.nhead, self.hidden_layers_dim, self.dropout_p,batch_first=True)
        self.encoder_layers = TransformerEncoderLayer(self.model_dim, self.nhead, self.hidden_layers_dim, self.dropout_p)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, self.nlayers)
        #self.init_weights()
        self.videoidx = torch.tensor(range(self.nvid)).to(device='cuda')
        self.frameidx = torch.tensor(range(self.nsegments)).to(device='cuda')


    def forward(self, text_embeddings, video_embeddings, sent_num_batch, face_embeddings, face_masks, face_segment_ids, slots, slot_masks, video_embed):
        # Special tokens
        #ipdb.set_trace()
        
        # slots are padded with -1, replacing them with 0's
        #slots[slots == -1] = 0
        video_slots = copy.deepcopy(slots)
        video_slots[video_slots == -1] = 0
        # remove extra dimension in slot_masks
        text_masks = slot_masks[:,1:]

        '''
        special_tensors = torch.tensor(range(self.nvid)).to(device='cuda')
        special_tensors = special_tensors.repeat(self.batch_size, 1)
        special_tokens = self.spcl_embed(special_tensors)
        spl_tokens_masks = torch.ones((self.batch_size, special_tokens.shape[1])).to(device='cuda')
        '''
        
        # Text Tokens - Ti + ei
        #ipdb.set_trace()
        if(text_embeddings is not None):
            video_ids = video_embed(video_slots)
            text_embeddings = text_embeddings + video_ids
            #text_masks = torch.zeros((self.batch_size, self.nvid), dtype=torch.float32).to(device='cuda')
            #for i, sent_num in enumerate(sent_num_batch):
            #    text_masks[i, :sent_num] = 1

        #ipdb.set_trace()
        # Video Tokens - V_i + e_i + e_t
        # Video dimension - 16 x 5 x 5 x 512 --> bs x 25 x 512
        # e_t - video_ids - [0,0,0,0,0],[1,1,1,1,1] ....
        # e_i - segment_ids - [0,1,2,3,4],[0,1,2,3,4] ....
        #ipdb.set_trace()
        video_masks = torch.zeros((self.batch_size, self.nvid * 5), dtype=torch.float32).to(device='cuda')
        # [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        video_segments = torch.tensor([i for i in range(5)] * 5).repeat(self.batch_size, 1)
        # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
        video_ids = torch.tensor([i for i in range(5) for j in range(5)]).repeat(self.batch_size, 1)
        video_e_t = self.segment_embed(video_segments.to(device='cuda'))
        video_e_i = video_embed(video_ids.to(device='cuda'))
        video_embeddings = video_embeddings + video_e_i + video_e_t
        # To-Do : It could be possible that all 5 videos are not present in a given batch,
        # in this case, we need to add video_masks accordingly.
        #video_masks = torch.ones((self.batch_size, video_embeddings.shape[1])).to(device='cuda')
        for i, sent_num in enumerate(sent_num_batch):
            video_masks[i, :sent_num*5] = 1

        #ipdb.set_trace()
        # Face Tokens - F_i + e_i + e_t
        # Face dimension - 16 x 50 x 512
        #ipdb.set_trace()
        # Flattenin the face segments - 5 x 10 - 50
        face_segment_ids = face_segment_ids.reshape(face_segment_ids.shape[0], face_segment_ids.shape[1] * face_segment_ids.shape[2])
        face_e_t =  self.segment_embed(face_segment_ids.to(device='cuda'))
        face_vid_ids = torch.tensor([i for i in range(5) for j in range(10)]).repeat(self.batch_size, 1)
        face_e_i = video_embed(face_vid_ids.to(device='cuda'))
        face_embeddings = face_embeddings + face_e_i + face_e_t
        face_masks = face_masks.reshape(self.batch_size, face_masks.shape[1] * face_masks.shape[2])

        # Final combined input to encoder 
        # Spl token + Text + Video + Face
        # encoder_input = special_tokens + text_embeddings + video_embeddings + face_embeddings
        # ipdb> special_tokens.shape, text_embeddings.shape, video_embeddings.shape, face_embeddings.shape
        # (torch.Size([16, 5, 512]), torch.Size([16, 5, 512]), torch.Size([16, 25, 512]), torch.Size([16, 50, 512]))
        # encoder_input.shape --> # torch.Size([16, 85, 512])

        #ipdb.set_trace()
        #encoder_input = torch.cat((special_tokens, text_embeddings, video_embeddings, face_embeddings), dim=1)
        #input_masks = torch.cat((spl_tokens_masks, text_masks,  video_masks, face_masks), dim=1)

        #if(torch.isnan(video_embeddings).any()):
        #    ipdb.set_trace()
        
        #if(torch.isnan(face_embeddings).any()):
        #    ipdb.set_trace()
        #ipdb.set_trace()
        if(text_embeddings is not None):
            encoder_input = torch.cat((text_embeddings, video_embeddings, face_embeddings), dim=1)
            input_masks = torch.cat((text_masks, video_masks, face_masks), dim=1)
        else:
            encoder_input = torch.cat((video_embeddings, face_embeddings), dim=1)
            input_masks = torch.cat((video_masks, face_masks), dim=1)
        '''
        Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked positions. 
        If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will be unchanged. 
        If a BoolTensor is provided, positions with True are not allowed to attend while False values will be unchanged. 
        '''
        input_masks = ~input_masks.bool()
        # encoder_output.shape --> torch.Size([85, 16, 512])
        #encoder_output = self.transformer_encoder(encoder_input.transpose(0,1), src_key_padding_mask = input_masks)

        return encoder_input.transpose(0,1), input_masks