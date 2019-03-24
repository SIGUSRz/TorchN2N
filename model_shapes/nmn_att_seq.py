import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from const import use_cuda


class EncoderRNN(nn.Module):
    def __init__(self, input_size, lstm_dim, embed_dim_que, num_layers):
        super(EncoderRNN, self).__init__()
        self.lstm_dim = lstm_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embed_dim_que)
        self.lstm = nn.LSTM(embed_dim_que, lstm_dim)

    def forward(self, question_variable, seq_len_batch, hidden):
        embedded = self.embedding(question_variable)
        embedded = embedded.permute(1, 0, 2)
        outputs, hidden = self.lstm(embedded)
        return outputs, hidden, embedded

    def init(self, batch_size):
        result = Variable(torch.zeros(self.num_layers, batch_size, self.lstm_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hyper, nmn_dim, EOS_token):
        super(AttnDecoderRNN, self).__init__()
        self.hyper = hyper
        self.lstm_dim = self.hyper.lstm_dim
        self.nmn_dim = nmn_dim  # number of nmn modules, ready to nmn
        self.embed_dim_nmn = self.hyper.embed_dim_nmn
        self.dropout_p = self.hyper.decoder_dropout
        self.num_layers = self.hyper.num_layers
        self.max_decoder_len = self.hyper.T_decoder
        self.EOS_token = EOS_token

        self.token_embeding = nn.Embedding(1, self.embed_dim_nmn)
        self.embedding = nn.Embedding(self.nmn_dim, self.embed_dim_nmn)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.embed_dim_nmn, self.lstm_dim)
        self.out = nn.Linear(self.lstm_dim * 2, self.nmn_dim)
        self.encoderLinear = nn.Linear(self.lstm_dim, self.lstm_dim)
        self.decoderLinear = nn.Linear(self.lstm_dim, self.lstm_dim)
        self.attnLinear = nn.Linear(self.lstm_dim, 1)
        self.batch_size = 0
        self._weight_init()

    '''
        compute if a token is valid at current sequence
        decoding_state [N,3]
        assembler_w [3,output_size, 4 ]
        assembler_b [output_size, 4]
        output [N, output_size]
    '''

    def _weight_init(self):
        torch.nn.init.xavier_uniform(self.decoderLinear.weight)
        torch.nn.init.xavier_uniform(self.attnLinear.weight)
        torch.nn.init.constant(self.decoderLinear.bias, 0)
        torch.nn.init.constant(self.attnLinear.bias, 0)

    '''
        for a give state compute the lstm hidden layer, attention and predicted layers
        can handle the situation where seq_len is 1 or >1 (i.e., s=using groudtruth layout)

        input parameters :
            step: int, time step of decoder
            embed_token: [decoder_len, batch], decoder_len=1 for step-by-step decoder
            previous_hidden_state: (h_n, c_n), dimmension:both are (num_layers * num_directions, batch, hidden_size)
            encoder_outputs : outputs from LSTM in encoder[seq_len, batch, hidden_size * num_directions]
            encoder_lens: list of input sequence lengths
            decoding_state: the state used to decide valid tokens

        output parameters : 
            predicted_token: [decoder_len, batch]
            Att_weighted_text: batch,out_len,txt_embed_dim
            log_seq_prob: [batch]
            neg_entropy: [batch]
    '''

    def _decode_step(self, step, embed_token, prev_hidden,
                     encoder_outputs, encoder_lens, decoding_state,
                     layout_variable, sample_token):

        # Run LSTM to get decoder hidden state
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        lstm_dim = encoder_outputs.size(2)
        out_len = embed_token.size(0)

        output, hidden = self.lstm(embed_token, prev_hidden)
        # Use function in Eq(2) of the paper to compute attention
        # size encoder_outputs (seq_len,batch_size,hidden_size)==>
        # (out_len,seq_len,batch_size,hidden_size)
        encoder_outputs_expand = encoder_outputs.view(1, seq_len, batch_size, lstm_dim).expand(
            out_len, seq_len,
            batch_size,
            lstm_dim)
        encoder_transform = self.encoderLinear(encoder_outputs_expand)

        # size output (out_len,batch_size,hidden_size)
        output_expand = output.view(out_len, 1, batch_size, lstm_dim).expand(out_len, seq_len,
                                                                             batch_size,
                                                                             lstm_dim)
        output_transfrom = self.decoderLinear(output_expand)

        # raw_attention size (out_len,seq_len,batch_size)
        raw_attention = self.attnLinear(F.tanh(encoder_transform +
                                               output_transfrom)). \
            view(out_len, seq_len, batch_size)
        # Eq2
        # (out_len, seq_len, batch_size)==>(batch_size,out_len,seq_len)
        raw_attention = raw_attention.permute(2, 0, 1)

        # mask the end of the question
        if encoder_lens is not None:
            mask = np.ones((batch_size, out_len, seq_len))
            for i, v in enumerate(encoder_lens):
                mask[i, :, 0:v] = 0
            mask_tensor = torch.ByteTensor(mask)
            mask_tensor = mask_tensor.cuda() if use_cuda else mask_tensor
            raw_attention.data.masked_fill_(mask_tensor, -float('inf'))
        attention = F.softmax(raw_attention, dim=2)  # (batch,out_len,seq_len)

        #  (seq_len,batch_size,hidden_size) ==>(batch_size,seq_len,hidden_size)
        encoder_batch_first = encoder_outputs.permute(1, 0, 2)
        context = torch.bmm(attention, encoder_batch_first)

        # (out_len,batch,hidden_size) --> (batch,out_len,hidden_size)
        output_batch_first = output.permute(1, 0, 2)

        # (batch,out_len,hidden_size*2)
        combined = torch.cat((context, output_batch_first), dim=2).permute(1, 0, 2)
        # print(combined.size())
        # print("Done\n\n\n")

        # [out_len,batch,out_size]
        output_prob = F.softmax(self.out(combined), dim=2)

        # probs
        probs = output_prob.view(-1, self.nmn_dim)
        probs_sum = torch.sum(probs, dim=1, keepdim=True)
        probs = probs / probs_sum

        if layout_variable is not None:
            predicted_token = layout_variable[:, step].view(-1, 1)
        elif sample_token:
            predicted_token = probs.multinomial()
        else:
            predicted_token = torch.max(probs, dim=1)[1].view(-1, 1)

        ##[batch_size, self.output_size]
        tmp = torch.zeros(batch_size, self.nmn_dim)
        tmp = tmp.cuda() if use_cuda else tmp
        predicted_token_encoded = tmp.scatter_(1, predicted_token.data, 1.0)
        predicted_token_encoded = predicted_token_encoded.cuda() if use_cuda else \
            predicted_token_encoded

        token_neg_entropy = torch.sum(probs.detach() * torch.log(probs + 0.000001), dim=1)

        ## compute log_seq_prob
        selected_token_log_prob = torch.log(
            torch.sum(probs * Variable(predicted_token_encoded), dim=1) + 0.000001)

        return predicted_token.permute(1, 0), hidden, attention, \
               token_neg_entropy, selected_token_log_prob

    def forward(self, encoder_hidden, encoder_outputs, encoder_lens,
                layout_variable, sample_token):
        self.batch_size = encoder_outputs.size(1)
        total_neg_entropy = 0
        total_seq_prob = 0

        # initiate step:
        step = 0
        start_token_variable = Variable(torch.LongTensor(np.zeros((1, self.batch_size))),
                                        requires_grad=False)
        start_token_variable = start_token_variable.cuda() if use_cuda else start_token_variable
        embed_token = self.token_embeding(start_token_variable)

        next_state = torch.FloatTensor([[0, 0, self.max_decoder_len]]).expand(
            self.batch_size, 3).contiguous()

        next_state = next_state.cuda() if use_cuda else next_state
        while step < self.max_decoder_len:
            predicted_token, previous_hidden, context, neg_entropy, log_seq_prob = \
                self._decode_step(step=step,
                                  embed_token=embed_token,
                                  prev_hidden=encoder_hidden,
                                  encoder_outputs=encoder_outputs,
                                  encoder_lens=encoder_lens,
                                  decoding_state=next_state,
                                  layout_variable=layout_variable,
                                  sample_token=sample_token)

            if step == 0:
                predicted_tokens = predicted_token
                total_neg_entropy = neg_entropy
                total_seq_prob = log_seq_prob
                context_total = context
            else:
                predicted_tokens = torch.cat((predicted_tokens, predicted_token))
                total_neg_entropy += neg_entropy
                total_seq_prob += log_seq_prob
                context_total = torch.cat((context_total, context), dim=1)

            step += 1
            embed_token = self.embedding(predicted_token)
            loop_state = torch.ne(predicted_token, self.EOS_token).any()

        return predicted_tokens, context_total, total_neg_entropy, total_seq_prob


class Seq2SeqAtt(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqAtt, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, question_variable, seq_len_batch, layout_variable, sample_token):
        encoder_hidden = self.encoder.init(seq_len_batch.shape[0])
        encoder_outputs, encoder_hidden, embed_que = \
            self.encoder(question_variable, seq_len_batch, encoder_hidden)
        decoder_results, attention, neg_entropy, log_seq_prob = self.decoder(
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            encoder_lens=seq_len_batch,
            layout_variable=layout_variable,
            sample_token=sample_token
        )
        # using attention from decoder and txt_embedded from the encoder
        # to get the attention weighted text
        # txt_embedded [seq_len,batch,input_encoding_size]
        # attention [batch, out_len,seq_len]
        txt_embedded_perm = embed_que.permute(1, 0, 2)
        att_weighted_text = torch.bmm(attention, txt_embedded_perm)

        return decoder_results, att_weighted_text, neg_entropy, log_seq_prob
