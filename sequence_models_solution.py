# Author: Yiping Jin

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 5
HIDDEN_DIM = 6
CHAR_EMBEDDING_DIM = 3
CHAR_HIDDEN_DIM = 3

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
ix_to_tag = {0:"DET", 1:"NN", 2:"V"}

char_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        for char in word:
            if char not in char_to_ix:
                char_to_ix[char] = len(char_to_ix)
                
                
class CharLSTMTagger(nn.Module):

    def __init__(self, embedding_dim, char_hidden_dim, hidden_dim, vocab_size, tagset_size):
        super(CharLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_hidden_dim = char_hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim+char_hidden_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence, char_encodings):
        embeds = self.word_embeddings(sentence)
        # concatenate the word embedding with character embedding
        embeds = torch.cat((embeds, char_encodings), 1) 
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class CharEncoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(CharEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes character embeddings as inputs, 
        # and outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, word):
        embeds = self.char_embeddings(word)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(word), 1, -1), self.hidden)
        return lstm_out[-1].view(1,-1)
        

char_encoder = CharEncoder(CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, len(char_to_ix))
model = CharLSTMTagger(EMBEDDING_DIM,CHAR_HIDDEN_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
full_model = nn.Sequential(char_encoder, model)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(full_model.parameters(), lr=0.1)

def _predict(sentence):
    """ helper method to prepare the input
    """
    inputs = prepare_sequence(sentence, word_to_ix)
    char_encodings = []
    for word in sentence:
        char_encoder.hidden = char_encoder.init_hidden()
        char_in = prepare_sequence(word, char_to_ix)
        char_encoding = char_encoder(char_in)
        char_encodings.append(char_encoding)
        
    char_encodings = torch.cat(char_encodings)
    return model(inputs, char_encodings)

print("=== Ground Truth: ===\n", training_data[0][0],'\n', training_data[0][1])    
# See what the scores are before training
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    sentence = training_data[0][0]
    tag_scores = _predict(sentence)
    _, max_indexes = torch.max(tag_scores, 1)
    print("=== Before training ===")
    print(tag_scores)
    print([ix_to_tag[int(max_id)] for max_id in max_indexes])

total_loss = 0
for epoch in range(400):  # again, normally you would NOT do 400 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        char_encoder.zero_grad()
        
        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)

        # Step 2.a encode char sequence
        char_encodings = []
        for word in sentence:
            char_encoder.hidden = char_encoder.init_hidden()
            char_in = prepare_sequence(word, char_to_ix)
            char_encoding = char_encoder(char_in)
            char_encodings.append(char_encoding)
        
        char_encodings = torch.cat(char_encodings)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in, char_encodings)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss
    if (epoch+1) % 100 == 0:
        print("average_loss epoch", epoch, ':', total_loss/100)
        total_loss = 0


# See what the scores are after training
with torch.no_grad():
    sentence = training_data[0][0]
    tag_scores = _predict(sentence)

    print("=== After training ===")
    print(tag_scores)
    _, max_indexes = torch.max(tag_scores, 1)
    print([ix_to_tag[int(max_id)] for max_id in max_indexes])