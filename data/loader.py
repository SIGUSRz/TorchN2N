import re
import json
from const import *
from torch.utils.data import Dataset
import torchvision.transforms as transforms

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


class VocabDict:
    def __init__(self, vocab_file):
        with open(vocab_file) as f:
            self.word_list = [s.strip() for s in f.readlines()]
        self.word2idx_dict = {w: i for i, w in enumerate(self.word_list)}
        self.num_vocab = len(self.word_list)
        self.unk_idx = self.word2idx_dict['<unk>'] if '<unk>' in self.word2idx_dict else None

    def idx2word(self, idx):
        return self.word_list[idx]

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.unk_idx is not None:
            return self.unk_idx
        else:
            raise ValueError('word %s is unknown' % w)

    def tokenize_and_index(self, sentence):
        idx_list = [self.word2idx(w) for w in self._tokenize(sentence)]
        return idx_list

    def _tokenize(self, sentence):
        tokens = SENTENCE_SPLIT_REGEX.split(sentence.lower())
        tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
        return tokens


class VQAset(Dataset):
    def __init__(self, data_path, hyper, assembler):
        self.hyper = hyper
        training_questions = []
        training_labels = []
        training_images_list = []
        gt_layout_list = []
        for image_set in image_sets:
            with open(data_path + training_text_files % image_set) as f:
                training_questions += [l.strip() for l in f.readlines()]
            with open(data_path + training_label_files % image_set) as f:
                training_labels += [l.strip() == 'true' for l in f.readlines()]
            training_images_list.append(np.load(data_path + training_image_files % image_set))
            with open(data_path + training_gt_layout_file % image_set) as f:
                gt_layout_list += json.load(f)
        self.image_array = np.concatenate(training_images_list)
        self.num_samples = len(training_questions)
        self.vocab = VocabDict(data_path + vocab_question_file)
        self.question_array = np.zeros((self.hyper.T_encoder, self.num_samples), np.int32)
        self.seq_length_array = np.zeros(self.num_samples, np.int32)
        self.gt_layout_array = np.zeros((self.hyper.T_decoder, self.num_samples), np.int32)
        self.label_array = np.zeros(self.num_samples, np.int32)
        self.transform = transforms.ToTensor()
        for pt in range(self.num_samples):
            tokens = training_questions[pt].split()
            self.seq_length_array[pt] = len(tokens)
            for t in range(len(tokens)):
                self.question_array[t, pt] = self.vocab.word2idx_dict[tokens[t]]
            self.label_array[pt] = training_labels[pt]
            self.gt_layout_array[:, pt] = assembler.module_list2tokens(
                gt_layout_list[pt], self.hyper.T_decoder)

    def __getitem__(self, idx):
        image = self.transform(self.image_array[idx, :, :, :])
        seq_len = self.seq_length_array[idx]
        gt_layout = self.gt_layout_array[:, idx]
        label = self.label_array[idx]
        input_seq = np.zeros(self.hyper.T_encoder, np.int32)
        input_seq[:seq_len] = self.question_array[:seq_len, idx]
        return image, seq_len, gt_layout, label, input_seq

    def __len__(self):
        return self.num_samples


