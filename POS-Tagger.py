import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from torchtext import data
from torchtext.legacy import data
import torch.optim as optim
import torch.nn.functional as F
import torchtext.vocab as torch_vocab

import math
from math import log, isfinite
from collections import Counter
import sys, os, time, platform, nltk, random


GLOVE_PATH = r".\glove.6B.100d.txt"
TRAIN_PATH = r".\en-ud-train.upos.tsv"

# GPU cuda cores will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def use_seed(seed = 2512021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_deterministic(True)


def read_annotated_sentence(f):
    '''returns (word, tag) list for an input file (f)'''
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n" and line.strip() != ""):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append( (word, tag) )
        line = f.readline()
    return sentence


def load_annotated_corpus(filename):
    '''returns list of sentences, each in a format of (word, tag) as returned by read_annotated_sentence;
     this input data will be used to calculate allTagCounts, perWordTagCounts, transitionCounts, emissionCounts'''
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            append = True
            for (w, t) in sentence:
                if ('http' in w) or (len(sentence) == 1):
                    append = False
                    break
            if append:
                sentences.append(sentence)
            sentence = read_annotated_sentence(f)

    return sentences


DUMMY = "<DUMMY>"
START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"

allTagCounts = Counter()
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
# missing Counter entries default to 0, not log(0)
A = {}  # transitions probabilities
B = {}  # emissions probabilities


def learn_params(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
    and emissionCounts data-structures.

    allTagCounts and perWordTagCounts - baseline tagging.
    A and B should are the log-probability of the normalized counts,
    based on transitionCounts & emissionCounts

    Args:
      tagged_sentences: a list of tagged sentences as returned by load_annotated_corpus().

    Return:
      a list containing [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B]
    """
    for sentence in tagged_sentences:
        previous_tag = START
        allTagCounts[START] += 1
        for word, tag in sentence:
            allTagCounts[tag] += 1

            if not perWordTagCounts.get(word):
                perWordTagCounts[word] = Counter()
            perWordTagCounts[word][tag] += 1

            if not emissionCounts.get(tag):
                emissionCounts[tag] = Counter()
            emissionCounts[tag][word] += 1

            if not transitionCounts.get(previous_tag):
                transitionCounts[previous_tag] = Counter()
            transitionCounts[previous_tag][tag] += 1

            previous_tag = tag

        if not transitionCounts.get(previous_tag):
            transitionCounts[previous_tag] = Counter()

        transitionCounts[previous_tag][END] += 1
        allTagCounts[END] += 1

    for previous_tag, tags_dict in transitionCounts.items():
        if not A.get(previous_tag):
            A[previous_tag] = dict()
        for tag, count in tags_dict.items():
            A[previous_tag][tag] = math.log(count / allTagCounts[previous_tag])

    for tag, observations_dict in emissionCounts.items():
        if not B.get(tag):
            B[tag] = dict()
        for observation, count in observations_dict.items():
            B[tag][observation] = math.log(count / allTagCounts[tag])

    return [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B]


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        Return:
        list: list of pairs
    """

    tagged_sentence = list()

    for word in sentence:
        if perWordTagCounts.get(word):
            tags_counts_dict = perWordTagCounts[word]
            tag = max(tags_counts_dict, key=tags_counts_dict.get)
        else:
            tag = max(allTagCounts, key=allTagCounts.get)

        tagged_sentence.append((word, tag))

    return tagged_sentence


#===========================================
#       POS tagging with HMM
#===========================================

def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterbi
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): the HMM Transition probabilities
        B (dict): the HMM emission probabilities.

    Return:
        list: list of pairs
    """
    v_last = viterbi(sentence, A, B)
    tags = retrace(v_last)
    tagged_sentence = list(zip(sentence[1:-1], tags[1:-1]))
    return tagged_sentence


def viterbi(sentence, A, B):
    """Creates the Viterbi matrix. Each column is a list of
    tuples representing cells. Each cell is a tuple (tag, ref_best_item_prev_position, prob_current),
    were t - tag at current position, r - reference to the best item from the previous position,
    p - log probability of the sequence so far.

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM transition probabilities
        B (dict): the HMM emission probabilities.

    Return:
        obj: the last item, tagged with END. should allow back-tracking.

        """

    sentence.insert(0, DUMMY)
    sentence.append(DUMMY)

    vit = {w_i: dict() for w_i, w in enumerate(sentence)}

    for word_ix, word in enumerate(sentence):
        if word_ix == 0:
            vit[word_ix][0] = [0, 0, 0]
            vit[word_ix][0][0] = START
            vit[word_ix][0][1] = None
            vit[word_ix][0][2] = 0

        elif word_ix == len(sentence) - 1:
            vit[word_ix][0] = [0, 0, 0]
            vit[word_ix][0][0] = END
            prev_best_ix, p = get_best_previous_state(vit[word_ix - 1], word, END)
            vit[word_ix][0][1] = prev_best_ix
            vit[word_ix][0][2] = p

        else:
            if word in perWordTagCounts.keys():
                tagset = [list(tags.keys()) for w, tags in perWordTagCounts.items() if w == word][0]
            else:
                tagset = list(allTagCounts.keys())

            for tag_ix, tag in enumerate(tagset):
                vit[word_ix][tag_ix] = [0, 0, 0]
                vit[word_ix][tag_ix][0] = tag
                prev_best_ix, p = get_best_previous_state(vit[word_ix - 1], word, tag)
                vit[word_ix][tag_ix][1] = prev_best_ix  # vit[word_ix - 1][prev_best_ix]
                vit[word_ix][tag_ix][2] = p

    v_last = get_last_best(vit, len(sentence) - 1)

    return v_last


def get_last_best(vit, word_ix):
    if not word_ix:
        return ('<DUMMY_START_TAG>', None, 0.0)
    else:
        probabilities = [cell[2] for tag_ix, cell in vit[word_ix].items()]
        max_p_ix = probabilities.index(max(probabilities))
        max_p = max(probabilities)
        max_p_tag = vit[word_ix][max_p_ix][0]
        return (max_p_tag, get_last_best(vit, word_ix - 1), max_p)


def get_best_previous_state(vit_prev_word, curr_word, curr_tag):
    '''get the best item for the previous state'''
    probabilities = [0] * len(vit_prev_word)
    prev_tags     = [0] * len(vit_prev_word)

    for ix, previous_item in enumerate(vit_prev_word.values()):
        prev_tags[ix] = previous_item[0]
        try:
            A_prev_t_curr_t = A[previous_item[0]][curr_tag]
        except:
            # transition from prev_t to curr_t doesn't exist
            A_prev_t_curr_t = math.log(1 / (allTagCounts[previous_item[0]] + len(allTagCounts)))
        try:
            B_curr_t_curr_w = B[curr_tag][curr_word]
        except:
            # emission from curr_w to curr_t doesn't exist
            B_curr_t_curr_w = math.log(1 / (allTagCounts[curr_tag] + len(allTagCounts)))
        prev_w_best_p = previous_item[2]
        probabilities[ix] = A_prev_t_curr_t + B_curr_t_curr_w + prev_w_best_p

    prev_best_ix = probabilities.index(max(probabilities))
    curr_best_p = probabilities[prev_best_ix]

    return prev_best_ix, curr_best_p


def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). Corresponds to the list of words in the sentence.
    """
    def reversed_retrace(end_item):
        if end_item[0] == START:
            return [START]
        else:
            return [end_item[0]] + reversed_retrace(end_item[1])
    reversed_tags = reversed_retrace(end_item)
    return list(reversed(reversed_tags))


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM transition probabilities
         B (dict): the HMM emission probabilities.
     """
    s = sentence.copy()
    s.insert(0, (START, START))
    s.append((END, END))

    p = 0   # joint log prob. of words and tags

    sentence = [(w, t) if w in perWordTagCounts.keys() else (UNK, t) for w, t in sentence]

    previous_tag = sentence[0]

    for i, (word, tag) in enumerate(sentence[1:]):
        try:
            p_transition = A[previous_tag][tag]
        except:
            p_transition = math.log(1 / (allTagCounts[previous_tag] + len(allTagCounts)))
        try:
            p_emission = B[tag][word]
        except:
            p_emission = math.log(1 / (allTagCounts[tag] + len(allTagCounts)))
        if not i:
            p = (p_transition + p_emission)
        else:
            p += (p_transition + p_emission)
        previous_tag = tag

    assert isfinite(p) and p < 0
    return p


#===========================================
#       POS tagging with BiLSTM
#===========================================

""" Two types of bi-LSTM models:
    1. a vanilla biLSTM in which the input layer is based on word embeddings
    2. a case-based BiLSTM in which input vectors combine a 3-dim binary vector
        encoding case information, see https://arxiv.org/pdf/1510.06168.pdf
"""

class CaseBiLSTMTagger(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, tagset_dim, num_layers, dropout, padding_idx, text, input_rep):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = tagset_dim
        self.num_layers = num_layers
        self.text = text
        self.input_rep = input_rep

        self.case_embedding = nn.Embedding(input_dim, 3)
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim+3, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.hidden2tag = nn.Linear(in_features=hidden_dim * 2, out_features=tagset_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentences, case):
        embed_out = self.embedding(sentences)
        case_embed_out = self.case_embedding(case)
        full_embed_out = torch.cat([embed_out, case_embed_out], dim=2)
        lstm_out, (hidden_states, cell_states) = self.lstm(full_embed_out)
        # print("lstm out shape: ", lstm_out.shape)
        tags_predictions = self.hidden2tag(self.dropout(lstm_out))
        # print("tags_predictions shape: ", tags_predictions.shape)
        return tags_predictions


class BiLSTMTagger(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, tagset_dim, num_layers, dropout, padding_idx, text, input_rep):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = tagset_dim
        self.num_layers = num_layers
        self.text = text
        self.input_rep = input_rep

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.hidden2tag = nn.Linear(in_features=hidden_dim * 2, out_features=tagset_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentences):
        embed_out = self.dropout(self.embedding(sentences))
        lstm_out, (hidden_states, cell_states) = self.lstm(embed_out)
        # print("lstm out shape: ", lstm_out.shape)
        tags_predictions = self.hidden2tag(self.dropout(lstm_out))
        # print("tags_predictions shape: ", tags_predictions.shape)
        return tags_predictions


def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)


def initialize_rnn_model(params_d):
    """Returns a dictionary with the objects and parameters needed to run/train_rnn
       the lstm model. The LSTM is initialized based on the specified parameters.

    Args:
        params_d (dict): a dictionary of parameters specifying the model. The dict
                        includes the following keys:
                        {'max_vocab_size': max vocabulary size (int),
                        'min_frequency': the occurrence threshold to consider (int),
                        'input_rep': 0 for the vanilla and 1 for the case-base (int),
                        'embedding_dimension': embedding vectors size (int),
                        'num_of_layers': number of layers (int),
                        'output_dimension': number of tags in tagset (int),
                        'pretrained_embeddings_fn': str,
                        'data_fn': str
                        }
    Return:
        a dictionary with at least the following key-value pairs:
                                       {'lstm': torch.nn.Module object,
                                       input_rep: [0|1]}
    """

    annotated_train = load_annotated_corpus(params_d["data_fn"])
    vocab = convert_data_to_csv(annotated_train, 'train.csv', input_rep=params_d["input_rep"]) # vocab is not lowercase

    vectors = load_pretrained_embeddings(params_d["pretrained_embeddings_fn"], vocab)

    text, tag, case = create_vocab(len(vocab),
                                   vectors,
                                   params_d["max_vocab_size"],
                                   params_d["min_frequency"],
                                   params_d["input_rep"])

    model = dict()
    if params_d["input_rep"]:
        lstm_model = CaseBiLSTMTagger(input_dim=len(text.vocab),
                                      embedding_dim=params_d["embedding_dimension"],
                                      hidden_dim=params_d["hidden_dimension"],
                                      tagset_dim=params_d["output_dimension"],
                                      num_layers=params_d["num_of_layers"],
                                      dropout=params_d["dropout"],
                                      padding_idx=text.vocab.stoi[text.pad_token],
                                      text=text,
                                      input_rep=params_d["input_rep"])
    else:
        lstm_model = BiLSTMTagger(input_dim=len(text.vocab),
                                  embedding_dim=params_d["embedding_dimension"],
                                  hidden_dim=params_d["hidden_dimension"],
                                  tagset_dim=params_d["output_dimension"],
                                  num_layers=params_d["num_of_layers"],
                                  dropout=params_d["dropout"],
                                  padding_idx=text.vocab.stoi[text.pad_token],
                                  text=text,
                                  input_rep=params_d["input_rep"])
    lstm_model.apply(init_weights)

    model["lstm"] = lstm_model
    model["input_rep"] = params_d["input_rep"]
    model["vectors"] = vectors
    model["text"] = text
    model["tag"] = tag
    model["case"] = case
    model["pad_ix"] = text.vocab.stoi[text.pad_token]
    model["batch_size"] = params_d["batch_size"]
    if params_d.get("n_epochs"):
        model["n_epochs"] = params_d["n_epochs"]
    else:
        model["n_epochs"] = 1

    return model


#no need for this one as part of the API
def get_model_params(model):
    """Returns a dictionary specifying the parameters of the specified model.
    This dictionary should be used to create another instance of the model.

    Args:
        model (torch.nn.Module): the network architecture

    Return:
        a dictionary, containing at least the following keys:
        {'input_dimension': int,
        'embedding_dimension': int,
        'num_of_layers': int,
        'output_dimension': int}
    """
    params_d = dict()
    params_d["input_dimension"] = model.input_dim
    params_d["hidden_dimension"] = model.hidden_dim
    params_d["output_dimension"] = model.output_dim
    params_d["num_of_layers"] = model.num_layers
    params_d["max_vocab_size"] = -1
    params_d["min_frequency"] = 1
    params_d["input_rep"] = model.input_rep
    params_d["embedding_dimension"] = model.embedding_dim
    params_d["dropout"] = 0.9

    return params_d


def load_pretrained_embeddings(path, vocab=None):
    """ Returns an object with the the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        (for efficiency (time and memory) - load only the vectors you need)

    Args:
        path (str): full path to the embeddings file
        vocab (list): a list of words to have embeddings for. Defaults to None.

    """
    vocab_lowercase = [w.lower() for w in vocab]
    vectors = dict()
    with open(path, "r", encoding='utf-8') as f:
        lines = f.readlines()

    for l in lines:
        word = l.split()[0]
        if word in vocab_lowercase:
            vector = [float(v) for v in l.split()[1:]]
            vectors[word] = vector

    convert_vecs_to_txt("embedding_vectors.txt", vectors)

    torch_vectors = torch_vocab.Vectors(name="./embedding_vectors.txt",
                                        unk_init=torch.Tensor.normal_,
                                        cache="embedding_vectors")

    return torch_vectors


def convert_vecs_to_txt(fn, vectors):
    with open(fn, "w", encoding="utf-8") as f:
        for word, ix in vectors.items():
            vector = " ".join([str(v) for v in vectors[word]])
            l = word + " " + vector + "\n"
            f.write(l)


def encode_w_case(w):
    '''returns an string token encoding the case of the word'''
    if w.islower():
        cb = "100"
    elif w.isupper():
        cb = "010"
    elif w == "<unk>":
        cb = "000"
    else:
        cb = "001"
    return cb


def convert_data_to_csv(annotated_data, fn=None, input_rep=None):
    if not annotated_data:
        return None

    sentences = list()
    tags = list()
    cases = list()
    vocab = set()

    for s in annotated_data:
        s_words = list()
        s_tags = list()
        s_case = list()
        for (w, t) in s:
            s_words.append(w.lower())
            s_tags.append(t)
            s_case.append(encode_w_case(w))
            if input_rep:
                vocab.add(w)
            else:
                vocab.add(w.lower())

        sentences.append(' '.join(s_words))
        tags.append(' '.join(s_tags))
        cases.append(' '.join(s_case))

    df = pd.DataFrame(data={'text': sentences, 'tag': tags, 'case': cases})

    df.to_csv(fn)
    return list(vocab)


def create_vocab(vocab_len, embed_vectors, max_vocab_size, min_frequency=None, input_rep=None):
    text = data.Field(lower=False)
    tag = data.Field(unk_token=None)
    case = data.Field(unk_token=None)

    fields = [(None, None), ('text', text), ('tag', tag), ('case', case)]
    train_tab = data.TabularDataset(path='./train.csv',
                                    format='csv',
                                    fields=fields,
                                    skip_header=False)

    if max_vocab_size == -1:
        text.build_vocab(train_tab,
                         min_freq=min_frequency,
                         vectors=embed_vectors,
                         unk_init=torch.Tensor.normal_
                         )

    elif max_vocab_size < vocab_len:
        text.build_vocab(train_tab,
                     max_size=max_vocab_size,
                     vectors=embed_vectors,
                         unk_init=torch.Tensor.normal_)
    else:
        text.build_vocab(train_tab,
                         vectors=embed_vectors,
                         unk_init=torch.Tensor.normal_)

    text.vocab.load_vectors(embed_vectors)
    tag.build_vocab(train_tab)
    case.build_vocab(train_tab)

    return text, tag, case


def get_accuracy(preds, y_tags, pad_idx):
    preds_argmax = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    relevant_preds = (y_tags != pad_idx).nonzero()
    correct = preds_argmax[relevant_preds].squeeze(1).eq(y_tags[relevant_preds])
    return correct.sum() / torch.FloatTensor([y_tags[relevant_preds].shape[0]])


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    return elapsed_mins


def evaluate(model, input_rep, iter, criterion, pad_ix):
    ep_loss, ep_acc = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in iter:
            text = batch.text
            text = text.to(device)
            tags = batch.tag
            tags = tags.to(device)
            if input_rep:
                case = batch.case
                preds = model(text, case)
            else:
                preds = model(text)
            preds = preds.view(-1, preds.shape[-1])
            tags = tags.view(-1)

            loss = criterion(preds, tags)
            acc = get_accuracy(preds, tags, pad_ix)

            ep_loss += loss.item()
            ep_acc += acc.item()
    return ep_loss / len(iter), ep_acc / len(iter)


def train(model, input_rep, iter, opt, criterion, pad_ix):
    ep_loss, ep_acc = 0, 0
    model.train()
    for batch in iter:
        text = batch.text
        tags = batch.tag
        opt.zero_grad()

        if input_rep:
            case = batch.case
            preds = model(text, case)
        else:
            preds = model(text)

        preds = preds.view(-1, preds.shape[-1])
        tags = tags.view(-1)
        loss = criterion(preds, tags)
        acc = get_accuracy(preds, tags, pad_ix)
        loss.backward()
        opt.step()
        ep_loss += loss.item()
        ep_acc += acc.item()

    return ep_loss / len(iter), ep_acc / len(iter)


def train_rnn(model, train_data, val_data=None):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (dict): the model dict as returned by initialize_rnn_model()
        train_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus()
        val_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus() to be used for validation.
                            Defaults to None
        input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                         1: case-base; <other int>: other models, if you are playful
    """

    lstm_model = model["lstm"]
    lstm_model.embedding.weight.data.copy_(model["text"].vocab.vectors)
    lstm_model.embedding.weight.data[model["text"].vocab.stoi[model["text"].pad_token]] = torch.zeros(lstm_model.embedding_dim)
    lstm_model.embedding.weight.requires_grad = False

    pad_ix = model["pad_ix"]
    BATCH_SIZE = model["batch_size"]

    if model.get("n_epochs"):
        N_EPOCHS = model["n_epochs"]
    else:
        N_EPOCHS = 10

    fields = [(None, None), ('text', model['text']), ('tag', model['tag']), ('case', model['case'])]

    train_tab = data.TabularDataset(path='./train.csv',
                                    format='csv',
                                    fields=fields,
                                    skip_header=True)
    if not val_data:
        train_tab, val_tab = train_tab.split(split_ratio=0.7)
    else:
        convert_data_to_csv(val_data, 'validation.csv', model['input_rep'])
        val_tab = data.TabularDataset(path='./validation.csv',
                                      format='csv',
                                      fields=fields,
                                      skip_header=True)

    train_iter, valid_iter = data.BucketIterator.splits((train_tab, val_tab),
                                                        sort_key=lambda x: len(x.text),
                                                        batch_size=BATCH_SIZE,
                                                        sort_within_batch=True)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_ix)
    criterion = criterion.to(device)

    optimizer = optim.Adam(lstm_model.parameters())

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start = time.time()
        train_loss, train_acc = train(lstm_model, model['input_rep'],train_iter, optimizer, criterion, pad_ix)
        valid_loss, valid_acc = evaluate(lstm_model, model['input_rep'], valid_iter, criterion, pad_ix)
        end = time.time()
        epoch_mins = epoch_time(start, end)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(lstm_model.state_dict(), 'model.pt')
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Valid. Loss: {valid_loss:.3f} | Valid. Acc: {valid_acc*100:.2f}%')


def rnn_tag_sentence(sentence, model):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
        (same index) in the input sentence and t is the predicted tag.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict):  a dictionary with the trained BiLSTM model and all that is needed
                        to tag a sentence.

    Return:
        list: list of pairs
    """
    tokens = [w.lower() for w in sentence]
    cases = [encode_w_case(w) for w in sentence]

    lstm_model = model['lstm']
    text = model['text']
    tag = model['tag']
    case = model['case']

    token_ixs = [text.vocab.stoi[t] for t in tokens]
    token_ixs_tensor = torch.LongTensor(token_ixs)

    token_ixs_tensor = token_ixs_tensor.unsqueeze(-1).to(device)
    lstm_model.eval()
    with torch.no_grad():
        if model["input_rep"]:
            cases_ixs = [case.vocab.stoi[c] for c in cases]
            cases_ixs_tensor = torch.LongTensor(cases_ixs)
            cases_ixs_tensor = cases_ixs_tensor.unsqueeze(-1).to(device)
            preds = lstm_model(token_ixs_tensor, cases_ixs_tensor)
        else:
            preds = lstm_model(token_ixs_tensor)

    preds_argmax = preds.argmax(-1)
    preds_tags = [tag.vocab.itos[t.item()] for t in preds_argmax]
    tagged_sentence = [(w, t) for w, t in zip(sentence, preds_tags)]

    return tagged_sentence


def get_best_performing_model_params():
    """Returns a dictionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """
    model_params = {'max_vocab_size': -1,
                    'min_frequency': 1,
                    'input_rep': 0,
                    'embedding_dimension': 100,
                    'num_of_layers': 2,
                    'output_dimension': 18,
                    'pretrained_embeddings_fn': "glove.6B.100d.txt",
                    'data_fn': "en-ud-train.upos.tsv",
                    "hidden_dimension": 256,
                    "dropout": 0.1,
                    "n_epochs": 8,
                    "batch_size": 64
                    }

    return model_params


#===========================================================
#       Wrapper function (tagging with a specified model)
#===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
           an ordered list of the parameters of the trained model (baseline, HMM)
           or the model itself and the input_rep flag (LSTMs).

        Models that must be supported:
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}
        3. Vanilla BiLSTM: {'blstm':[model_dict]}
        4. BiLSTM+case: {'cblstm': [model_dict]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

        Parameters for an LSTM: the model dictionary (allows tagging the given sentence)

    Return:
        list: list of pairs
    """
    if list(model.keys())[0] == 'baseline':
        return baseline_tag_sentence(sentence, list(model.values())[0][0], list(model.values())[0][1])
    if list(model.keys())[0] == 'hmm':
        return hmm_tag_sentence(sentence, list(model.values())[0][0], list(model.values())[0][1])
    if list(model.keys())[0] == 'blstm':
        return rnn_tag_sentence(sentence, list(model.values())[0][0])
    if list(model.keys())[0] == 'cblstm':
        return rnn_tag_sentence(sentence, list(model.values())[0][0])


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correctly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger
    """
    assert len(gold_sentence) == len(pred_sentence)

    OOV = 0
    correct = 0
    correctOOV = 0

    for i, (word, gold_tag) in enumerate(gold_sentence):
        pred_tag = pred_sentence[i][1]
        if word not in perWordTagCounts.keys():
            OOV += 1
        if gold_tag == pred_tag:
            if word not in perWordTagCounts.keys():
                correctOOV += 1
            else:
                correct += 1

    return correct, correctOOV, OOV