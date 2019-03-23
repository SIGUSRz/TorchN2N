import argparse
from const import *
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import data.loader as loader
from model_shapes.vision_model import CNN
from model_shapes.nmn_assembler import Assembler

from model_shapes.end2endModuleNet import *
from model_shapes.custom_loss import custom_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument("--model_type", type=str,
                        choices=[model_type_scratch,
                                 model_type_gt,
                                 model_type_gt_rl],
                        required=True,
                        help='models:' + model_type_scratch + ',' +
                             model_type_gt + ', ' +
                             model_type_gt_rl)
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    gpu_id = args.gpu_id  # set GPU id to use
    exp_name = args.model_type
    model_type = args.model_type
    data_path = args.data_path
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    os.makedirs("exp", exist_ok=True)
    log_path = os.path.join("exp", exp_name)
    if os.path.isdir(log_path):
        os.system("rm -rf %s" % log_path)
    os.makedirs(log_path)
    lambda_entropy = 0
    
    criterion_layout = custom_loss(lambda_entropy = lambda_entropy)
    criterion_answer = nn.CrossEntropyLoss(size_average=False,reduce=False)

    params = HyperParameter(model_type)
    assembler = Assembler(data_path + vocab_layout_file)
    dataset = loader.VQAset(data_path, params, assembler)
    data_loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        num_workers=2,
        shuffle=True
    ) 

    num_layers = 2
    T_decoder = 256
    
    num_vocab_nmn = len(assembler.module_names)
    #dataset.vocab.
    vision = CNN()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vision.parameters(), lr=params.learning_rate)
    
    myModel = end2endModuleNet(num_vocab_txt= dataset.vocab.num_vocab, num_vocab_nmn=num_vocab_nmn, out_num_choices = 2,
                               embed_dim_nmn=params.embed_dim_nmn, embed_dim_txt=params.embed_dim_txt,
                               image_height=params.H_im, image_width=params.W_im, in_image_dim=params.D_feat,
                               hidden_size=params.lstm_dim, assembler=assembler, layout_criterion=criterion_layout,
                               max_layout_len= T_decoder,
                               answer_criterion=criterion_answer, num_layers=num_layers, decoder_dropout=0)
                               
    myOptimizer = optim.Adam(myModel.parameters(), weight_decay=params.weight_decay, lr=params.learning_rate)


    avg_accuracy = 0
    accuracy_decay = 0.99
    avg_layout_accuracy = 0
    updated_baseline = np.log(28)




    for batch_idx, (image_batch, seq_len_batch, gt_layout_batch, label_batch, input_seq_batch) in \
            enumerate(data_loader):
        optimizer.zero_grad()
        print(image_batch.size())
        input_images = vision(image_batch)
        
        print(input_images.size())

        _, n_sample = input_seq_batch.shape
        input_text_seq_lens = seq_len_batch
        input_text_seqs = input_seq_batch
        input_layouts = gt_layout_batch
        input_answers = label_batch


        n_correct_layout = 0
        n_correct_answer = 0
        print(input_text_seqs)
        input_txt_variable = Variable(torch.LongTensor((input_text_seqs.long())))
        input_txt_variable = input_txt_variable.cuda() if use_cuda else input_txt_variable

        input_layout_variable = None

        if model_type == model_type_gt:
            input_layout_variable = Variable(torch.LongTensor((input_layouts.long())))
            input_layout_variable = input_layout_variable.cuda() if use_cuda else input_layout_variable
   
        myOptimizer.zero_grad()

        total_loss,avg_answer_loss ,myAnswer, predicted_layouts, expr_validity_array, updated_baseline \
            = myModel(input_txt_variable=input_txt_variable, input_text_seq_lens=input_text_seq_lens,
                    input_answers=input_answers, input_images=input_images,policy_gradient_baseline=updated_baseline,
                    baseline_decay=params.baseline_decay, input_layout_variable=input_layout_variable,
                  sample_token=params.decoder_sampling
                  )

        if total_loss is not None:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm(myModel.parameters(), max_grad_l2_norm)
        
        myOptimizer.step()

        layout_accuracy = np.mean(np.all(predicted_layouts == input_layouts, axis=0))
        avg_layout_accuracy += (1 - accuracy_decay) * (layout_accuracy - avg_layout_accuracy)

        accuracy = np.mean(np.logical_and(expr_validity_array, myAnswer == input_answers))
        avg_accuracy += (1 - accuracy_decay) * (accuracy - avg_accuracy)
        validity = np.mean(expr_validity_array)

        if (i_iter + 1) % 20 == 0 :
            print("iter:", i_iter + 1,
                " cur_layout_acc:%.3f"% layout_accuracy, " avg_layout_acc:%.3f"% avg_layout_accuracy,
                " cur_ans_acc:%.4f"% accuracy, " avg_answer_acc:%.4f"% avg_accuracy,
                "total loss:%.4f"%total_loss.data.cpu().numpy()[0],
                "avg_answer_loss:%.4f"% avg_answer_loss.data.cpu().numpy()[0])

            sys.stdout.flush()

    # Save snapshot
        if  (i_iter + 1) % snapshot_interval == 0 or (i_iter + 1) == max_iter:
            model_snapshot_file = os.path.join(snapshot_dir, "model_%08d" % (i_iter + 1))
            torch.save(myModel, model_snapshot_file)
            print('snapshot saved to ' + model_snapshot_file )
            sys.stdout.flush()








