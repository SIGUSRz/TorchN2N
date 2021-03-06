import torch
import torch.nn as nn
import sys
from model_shapes.nmn_att_seq import *
from model_shapes.module_net import *
from model_shapes.vision_model import CNN
from Utils.utils import unique_columns


class N2NModuleNet(nn.Module):
    def __init__(self, num_que_vocab, num_answer, hyper, assembler,
                 layout_criterion, answer_criterion):
        super(N2NModuleNet, self).__init__()

        self.assembler = assembler
        self.layout_criterion = layout_criterion
        self.answer_criterion = answer_criterion
        self.hyper = hyper

        # initiate encoder and decoder
        encoder = EncoderRNN(input_size=num_que_vocab,
                             lstm_dim=self.hyper.lstm_dim,
                             embed_dim_que=self.hyper.embed_dim_que,
                             num_layers=self.hyper.num_layers)
        decoder = AttnDecoderRNN(hyper=self.hyper,
                                 nmn_dim=self.assembler.num_module,
                                 EOS_token=self.assembler.EOS_idx)

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()

        # initatiate attentionSeq2seq
        seq_model = Seq2SeqAtt(encoder, decoder)
        self.seq_model = seq_model.cuda() if use_cuda else seq_model

        vision_model = CNN()
        self.vision_model = vision_model.cuda() if use_cuda else vision_model

        # initiate moduleNet
        mod_net = ModuleNet(image_height=self.hyper.H_im,
                            image_width=self.hyper.W_im,
                            image_channel=self.hyper.D_feat,
                            embed_dim_que=self.hyper.embed_dim_que,
                            num_answer=2,
                            lstm_dim=self.hyper.lstm_dim)

        self.mod_net = mod_net.cuda() if use_cuda else mod_net

    def forward(self, question_variable, layout_variable,
                image_batch, seq_len_batch, label_batch,
                sample_token, policy_gradient_baseline=None,
                baseline_decay=None):
        batch_size = seq_len_batch.shape[0]

        predicted_tokens, attentions, neg_entropy, log_seq_prob = \
            self.seq_model(question_variable, seq_len_batch, layout_variable,
                           sample_token)

        layout_loss = None
        if layout_variable is not None:
            layout_loss = torch.mean(-log_seq_prob)

        predicted_layouts = np.asarray(predicted_tokens.cpu().data.numpy())
        expr_list, expr_validity_array = self.assembler.assemble(predicted_layouts)

        # group samples based on layout
        sample_groups_by_layout = unique_columns(predicted_layouts)

        # run moduleNet
        answer_losses = None
        policy_gradient_losses = None
        avg_answer_loss = None
        total_loss = None
        updated_baseline = policy_gradient_baseline
        current_answer = np.zeros(batch_size)

        for sample_group in sample_groups_by_layout:
            if sample_group.shape == 0:
                continue

            first_in_group = sample_group[0]
            if expr_validity_array[first_in_group]:
                layout_exp = expr_list[first_in_group]

                ith_answer = label_batch[sample_group]
                ith_answer_variable = Variable(ith_answer.long())
                ith_answer_variable = ith_answer_variable.cuda() \
                    if use_cuda else ith_answer_variable
                que_attention = attentions[sample_group, :]

                ith_image = image_batch[sample_group, :, :, :]
                ith_images_variable = Variable(torch.FloatTensor(ith_image))
                ith_images_variable = ith_images_variable.cuda() \
                    if use_cuda else ith_images_variable

                # image[batch_size, H_feat, W_feat, D_feat] ==>
                # [batch_size, D_feat, W_feat, H_feat] for conv2d
                # ith_images_variable = ith_images_variable.permute(0, 3, 1, 2)

                ith_images_variable = ith_images_variable.contiguous()

                answers = self.mod_net(input_image_variable=ith_images_variable,
                                       input_text_attention_variable=que_attention,
                                       target_answer_variable=ith_answer_variable,
                                       expr_list=layout_exp)
                current_answer[sample_group] = torch.topk(answers, 1)[1].cpu().data.numpy()[:, 0]

                # compute loss function only when answer is provided
                if ith_answer_variable is not None:
                    current_answer_loss = self.answer_criterion(answers, ith_answer_variable)
                    sample_group_tensor = torch.cuda.LongTensor(
                        sample_group) if use_cuda else torch.LongTensor(sample_group)

                    current_log_seq_prob = log_seq_prob[sample_group_tensor]
                    current_answer_loss_val = Variable(current_answer_loss.data,
                                                       requires_grad=False)
                    tmp1 = current_answer_loss_val - policy_gradient_baseline
                    current_policy_gradient_loss = tmp1 * current_log_seq_prob

                    if answer_losses is None:
                        answer_losses = current_answer_loss
                        policy_gradient_losses = current_policy_gradient_loss
                    else:
                        answer_losses = torch.cat((answer_losses, current_answer_loss))
                        policy_gradient_losses = torch.cat(
                            (policy_gradient_losses, current_policy_gradient_loss))

        try:
            if label_batch is not None:
                total_loss, avg_answer_loss = \
                    self.layout_criterion(neg_entropy=neg_entropy,
                                          answer_loss=answer_losses,
                                          policy_gradient_losses=policy_gradient_losses,
                                          layout_loss=layout_loss)
                ##update layout policy baseline
                avg_sample_loss = torch.mean(answer_losses)
                avg_sample_loss_value = avg_sample_loss.cpu().data.numpy()[0]
                updated_baseline = policy_gradient_baseline + (1 - baseline_decay) * (
                        avg_sample_loss_value - policy_gradient_baseline)

        except:
            print("sample_group = ", sample_group)
            print("neg_entropy=", neg_entropy)
            print("answer_losses=", answer_losses)
            print("policy_gradient_losses=", policy_gradient_losses)
            print("layout_loss=", layout_loss)
            sys.stdout.flush()
            sys.exit("Exception Occur")

        return total_loss, avg_answer_loss, current_answer, predicted_layouts, expr_validity_array, updated_baseline
