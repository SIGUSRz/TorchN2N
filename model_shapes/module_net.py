import torch
import torch.nn as nn
import torch.nn.functional as F
from const import *
from model_shapes.modules import *
from torch.autograd import Variable


class ModuleNet(nn.Module):
    def __init__(self, image_height, image_width, image_channel, embed_dim_que, num_answer,
                 lstm_dim):
        super(ModuleNet, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.embed_dim_que = embed_dim_que
        self.num_answer = num_answer
        self.lstm_dim = lstm_dim
        self.FindModule = FindModule(image_dim=image_channel,
                                     text_dim=embed_dim_que,
                                     map_dim=lstm_dim)
        self.TransformModule = TransformModule(image_dim=image_channel,
                                               text_dim=embed_dim_que,
                                               map_dim=lstm_dim)
        self.AndModule = AndModule()
        self.FilterModule = FilterModule(findModule=self.FindModule, andModule=self.AndModule)

        self.CountModule = CountModule(output_num_choice=num_answer,
                                       image_height=image_height, image_width=image_width)

        self.layout2module = {
            '_Find': self.FindModule,
            '_Transform': self.TransformModule,
            '_And': self.AndModule,
            '_Filter': self.FilterModule,
            '_Answer': self.CountModule
        }

    # text[N, D_text]

    def recursively_assemble_network(self, input_image_variable, input_text_attention_variable,
                                     expr_list):
        current_module = self.layout2module[expr_list['module']]
        time_idx = expr_list['time_idx']
        text_index = Variable(torch.LongTensor([time_idx]))
        text_index = text_index.cuda() if use_cuda else text_index
        text_at_time = torch.index_select(input_text_attention_variable, dim=1,
                                          index=text_index).view(-1, self.embed_dim_que)

        input_0 = None
        input_1 = None

        if 'input_0' in expr_list:
            input_0 = self.recursively_assemble_network(input_image_variable,
                                                        input_text_attention_variable,
                                                        expr_list['input_0'])
        if 'input_1' in expr_list:
            input_1 = self.recursively_assemble_network(input_image_variable,
                                                        input_text_attention_variable,
                                                        expr_list['input_1'])

        res = current_module(input_image_variable, text_at_time, input_0, input_1)
        return res

    def forward(self, input_image_variable, input_text_attention_variable, target_answer_variable,
                expr_list):

        ##for now assume batch_size = 1
        result = self.recursively_assemble_network(input_image_variable,
                                                   input_text_attention_variable, expr_list)

        return result
