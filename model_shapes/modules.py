import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

'''
NOTE: in all modules, 
image_feat [N,D_image,H,W]
text [N,D_text]
attention [N,1,H,W]
'''


class FindModule(nn.Module):
    '''
    Mapping image_feat_grid X text_param ->att.grid
    (N,D_image,H,W) X (N,1,D_text) --> [N,1,H,W]
    '''

    def __init__(self, image_dim, text_dim, map_dim):
        super(FindModule, self).__init__()
        self.map_dim = map_dim
        self.conv1 = nn.Conv2d(image_dim, map_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(map_dim, 1, kernel_size=1)
        self.textfc = nn.Linear(text_dim, map_dim)

    def forward(self, input_image_feat, input_text, input_image_attention1=None,
                input_image_attention2=None):
        image_mapped = self.conv1(input_image_feat)  # (N, map_dim, H, W)
        text_mapped = self.textfc(input_text).view(-1, self.map_dim, 1, 1).expand_as(image_mapped)
        elmtwize_mult = image_mapped * text_mapped
        elmtwize_mult = F.normalize(elmtwize_mult, p=2, dim=1)  # (N, map_dim, H, W)
        att_grid = self.conv2(elmtwize_mult)  # (N, 1, H, W)
        return att_grid


class FilterModule(nn.Module):
    def __init__(self, findModule, andModule):
        super(FilterModule, self).__init__()
        self.andModule = andModule
        self.findModule = findModule

    def forward(self, input_image_feat, input_text, input_image_attention1=None,
                input_image_attention2=None):
        find_result = self.findModule(input_image_feat, input_text, input_image_attention1,
                                      input_image_attention2)
        att_grid = self.andModule(input_image_feat, input_text, input_image_attention1, find_result)
        return att_grid


class TransformModule(nn.Module):
    def __init__(self, image_dim, text_dim, map_dim, kernel_size=5, padding=2):
        super(TransformModule, self).__init__()
        self.map_dim = map_dim
        self.conv1 = nn.Conv2d(1, map_dim, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(map_dim, 1, kernel_size=1)
        self.textfc = nn.Linear(text_dim, map_dim)

    def forward(self, input_image_feat, input_text, input_image_attention1=None,
                input_image_attention2=None):
        image_att_mapped = self.conv1(input_image_attention1)  # (N, map_dim, H, W)
        text_mapped = self.textfc(input_text).view(-1, self.map_dim, 1, 1).expand_as(
            image_att_mapped)
        elmtwize_mult = image_att_mapped * text_mapped
        elmtwize_mult = F.normalize(elmtwize_mult, p=2, dim=1)  # (N, map_dim, H, W)
        att_grid = self.conv2(elmtwize_mult)  # (N, 1, H, W)
        return att_grid


class AndModule(nn.Module):
    def __init__(self):
        super(AndModule, self).__init__()

    def forward(self, input_image_feat, input_text, input_image_attention1=None,
                input_image_attention2=None):
        return torch.max(input_image_attention1, input_image_attention2)


class AnswerModule(nn.Module):
    def __init__(self, num_answer, image_height, image_width):
        super(AnswerModule, self).__init__()
        self.num_answer = num_answer
        self.lc_out = nn.Linear(image_height * image_width + 3, self.num_answer)

    def forward(self, image_feat, input_text, input_image_attention1=None,
                input_image_attention2=None):
        print(image_feat.size())
        print("Done\n\n\n")
        H, W = input_image_attention1.shape[2:4]
        att_all = input_image_attention1.view(-1, H * W)  ##flatten attention to [N, H*W]
        att_avg = torch.mean(att_all, 1, keepdim=True)
        att_min = torch.min(att_all, 1, keepdim=True)[0]
        att_max = torch.max(att_all, 1, keepdim=True)[0]
        att_concat = torch.cat((att_all, att_avg, att_min, att_max), 1)
        scores = self.lc_out(att_concat)
        return scores


class CountModule(nn.Module):
    def __init__(self, output_num_choice, image_height, image_width):
        super(CountModule, self).__init__()
        self.out_num_choice = output_num_choice
        self.lc_out = nn.Linear(image_height * image_width + 3, self.out_num_choice)

    def forward(self, input_image_feat, input_text, input_image_attention1=None,
                input_image_attention2=None):
        print(input_image_feat.size())
        print("Done\n\n\n")
        H, W = input_image_attention1.shape[2:4]
        att_all = input_image_attention1.view(-1, H * W)  ##flatten attention to [N, H*W]
        att_avg = torch.mean(att_all, 1, keepdim=True)
        att_min = torch.min(att_all, 1, keepdim=True)[0]
        att_max = torch.max(att_all, 1, keepdim=True)[0]
        att_concat = torch.cat((att_all, att_avg, att_min, att_max), 1)
        scores = self.lc_out(att_concat)
        return scores
