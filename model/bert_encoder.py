import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_model import base_model
from transformers import BertModel, BertConfig
from transformers import BertForMaskedLM 

class Bert_Encoder(base_model):

    def __init__(self, config):
        super(Bert_Encoder, self).__init__()

        # load model
        self.encoder = BertModel.from_pretrained(config.bert_path).cuda()
        self.bert_config = BertConfig.from_pretrained(config.bert_path)

        # the dimension for the final outputs
        self.output_size = config.encoder_output_size

        self.drop = nn.Dropout(config.drop_out)

        # find which encoding is used
        if config.pattern in ['standard', 'entity_marker']:
            self.pattern = config.pattern
        else:
            raise Exception('Wrong encoding.')
        config.hidden_size = self.bert_config.hidden_size
        config.output_size = config.encoder_output_size
        if self.pattern == 'entity_marker':
            self.encoder.resize_token_embeddings(config.vocab_size + config.marker_size)
            self.linear_transform = nn.Linear(self.bert_config.hidden_size * 2, self.output_size, bias=True)
        else:
            self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)

        self.layer_normalization = nn.LayerNorm([self.output_size])

    def get_output_size(self):
        return self.output_size

    def forward(self, inputs):
        '''
        :param inputs: of dimension [B, N]
        :return: a result of size [B, H*2] or [B, H], according to different strategy
        '''
        # generate representation under a certain encoding strategy
        if self.pattern == 'standard':
            # in the standard mode, the representation is generated according to
            #  the representation of[CLS] mark.
            output = self.encoder(inputs)[1]
        else:
            e11 = []
            e21 = []
            # for each sample in the batch, acquire the positions of its [E11] and [E21]
            for i in range(inputs.size()[0]):
                tokens = inputs[i].cpu().numpy()
                try: # hot fix for just 1 sample (error when test)
                    e21.append(np.argwhere(tokens == 30524)[0][0]) #
                except:
                    e21.append(0)
                e11.append(np.argwhere(tokens == 30522)[0][0])

            # input the sample to BERT
            tokens_output = self.encoder(inputs)[0] # [B,N] --> [B,N,H]
            output = []

            # for each sample in the batch, acquire its representations for [E11] and [E21]
            for i in range(len(e11)):
                instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())
                instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda())
                output.append(instance_output)  # [B,N] --> [B,2,H]

            # for each sample in the batch, concatenate the representations of [E11] and [E21], and reshape
            output = torch.cat(output, dim=0)
            output = output.view(output.size()[0], -1)  # [B,N] --> [B,H*2]
        return output
    
    
    
class Bert_EncoderMLM(base_model):

    def __init__(self, config):
        super(Bert_EncoderMLM, self).__init__()
        self.config = config
        # load model
        self.encoder = BertModel.from_pretrained(config.bert_path).to(config.device)
        self.lm_head = BertForMaskedLM.from_pretrained(config.bert_path).to(config.device).cls

        self.bert_word_embedding = self.encoder.get_input_embeddings()
        self.embedding_dim = self.bert_word_embedding.embedding_dim

        # the dimension for the final outputs
        self.drop = nn.Dropout(config.drop_out)
        self.output_size = self.embedding_dim
        
        # find which encoding is used
        if config.pattern in ['standard', 'entity_marker', 'entity_marker_mask','mask']:
            self.pattern = config.pattern
        else:
            raise Exception('Wrong encoding.')
        
        if self.pattern == 'mask':
            self.linear_transform = nn.Linear(self.embedding_dim,self.embedding_dim, bias=True).to(config.device)
            self.info_nce_fc = nn.Linear(config.vocab_size, self.embedding_dim) .to(config.device)   
        else:
            raise NotImplementedError('Only mask pattern is implemented')

        self.layer_normalization = nn.LayerNorm([self.output_size])
    def infoNCE_f(self, V, C):
        """
        V : B x vocab_size
        C : B x embedding_dim
        """
        try:
            out = self.info_nce_fc(V) # B x embedding_dim
            out = torch.matmul(out , C.t()) # B x B

        except:
            print("V.shape: ", V.shape)
            print("C.shape: ", C.shape)
            print("info_nce_fc: ", self.info_nce_fc)
        return out
    
    def get_output_size(self):
        return self.output_size

    def forward(self, inputs):
        '''
        :param inputs: of dimension [B, N]
        :return: a result of size [B, H*2] or [B, H], according to different strategy
        '''
        if self.pattern == 'mask':
            # in the standard mode, the representation is generated according to
            #  the representation of[CLS] mark.
            batch_size = inputs.shape[0]
            output = self.encoder(inputs)[0]
            
            #return [MASK] hidden state
            mask_pos = []
            for i in range(batch_size):
                tokens = inputs[i].cpu().numpy()
                try:
                    mask = np.argwhere(tokens == 103)[0][0]
                except:
                    print('No mask token found in the input sequence')
                    mask = 0
                mask_pos.append(mask)
            mask_hidden = output[torch.arange(batch_size), torch.tensor(mask_pos).to(self.config.device)]

            lmhead_output = self.lm_head(mask_hidden)
            return mask_hidden,lmhead_output
        else:
            raise NotImplementedError('Only mask pattern is implemented')
    


