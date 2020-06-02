###############################################################################
#######################  1. LOAD THE TRAINING TEXT  ###########################
###############################################################################
with open("data/reviews.txt") as f:
    reviews = f.read()
    
with open("data/labels.txt") as f:
    labels = f.read()
    
  
###############################################################################
##########################  2. TEXT PRE-PROCESSING  ###########################
###############################################################################
from string import punctuation

def preprocess(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in punctuation])
    all_reviews = text.split("\n")
    text = " ".join(text)
    all_words = text.split()
    
    return all_reviews, all_words


all_reviews, all_words = preprocess(reviews)


###############################################################################
##################  3. CREATE DICTIONARIES & ENCODE REVIEWS  ##################
###############################################################################
from collections import Counter

word_counts = Counter(all_words)
word_list = sorted(word_counts, keys = word_counts.get, reverse = True)
vocab_to_int = {word:idx+1 for idx, word in enumerate(word_list)}
int_to_vocab = {idx:word for word, idx in vocab_to_int.items()}
encoded_reviews = [[vocab_to_int[word] for word in review] for review in all_reviews]


###############################################################################
#############################  4. ENCODE LABELS ###############################
###############################################################################
all_labels = labels.split("\n")
encoded_labels = [1 if label == "positive" else 0 for label in all_labels]
assert len(encoded_reviews) == len(encoded_labels), "# of encoded reivews & encoded labels must be the same!"


###############################################################################
#####################  5. GET RID OF LENGTH-0 REVIEWS   #######################
###############################################################################
import numpy as np
import torch

encoded_labels = np.array( [label for idx, label in enumerate(encoded_labels) if len(encoded_reviews[idx]) > 0] )
encoded_reviews = [review for review in encoded_reviews if len(review) > 0]


###############################################################################
######################  6. MAKE ALL REVIEWS SAME LENGTH  #######################
###############################################################################
def pad_text(encoded_reviews, seq_length):
    
    reviews = []
    
    for review in encoded_reviews:
        if len(review) >= seq_length:
            reviews.append(review[:seq_length])
        else:
            reviews.append([0]*(seq_length-len(review)) + review)
        
    return np.array(reviews)


padded_reviews = pad_text(encoded_reviews, seq_length = 200)
        

###############################################################################
##############  7. SPLIT DATA & GET (REVIEW, LABEL) DATALOADER  ###############
###############################################################################
train_ratio = 0.8
valid_ratio = (1 - train_ratio)/2
total = padded_reviews.shape[0]
train_cutoff = int(total * train_ratio)
valid_cutoff = int(total * (1 - valid_ratio))

train_x, train_y = padded_reviews[:train_cutoff], encoded_labels[:train_cutoff]
valid_x, valid_y = padded_reviews[:train_cutoff : valid_cutoff], encoded_labels[train_cutoff : valid_cutoff]
test_x, test_y = padded_reviews[valid_cutoff:], encoded_labels[valid_cutoff:]

from torch.utils.data import TensorDataset, DataLoader

train_data = TensorDataset(train_x, train_y)
valid_data = TensorDataset(valid_x, valid_y)
test_data = TensorDataset(test_x, test_y)

batch_size = 50
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)


###############################################################################
#########################  8. DEFINE THE LSTM MODEL  ##########################
###############################################################################
from torch import nn

class SentimentLSTM(nn.Module):
    
    def __init__(self, n_vocab, n_embed, n_hidden, n_output, n_layers, drop_p = 0.5):
        super().__init__()
        # params: "n_" means dimension
        self.n_vocab = n_vocab     # number of unique words in vocabulary
        self.n_layers = n_layers   # number of LSTM layers 
        self.n_hidden = n_hidden   # number of hidden nodes in LSTM
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers, batch_first = True, dropout = drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward (self, input_words):
                                             # INPUT   :  (batch_size, seq_length)
        embedded_words = self.embedding(input_words)    # (batch_size, seq_length, n_embed)
        lstm_out, h = self.lstm(embedded_words)         # (batch_size, seq_length, n_hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, self.n_hidden) # (batch_size*seq_length, n_hidden)
        fc_out = self.fc(lstm_out)                      # (batch_size*seq_length, n_output)
        sigmoid_out = self.sigmoid(fc_out)              # (batch_size*seq_length, n_output)
        sigmoid_out = sigmoid_out.view(batch_size, -1)  # (batch_size, seq_length*n_output)
        
        # extract the output of ONLY the LAST output of the LAST element of the sequence
        sigmoid_last = sigmoid_out[:, -1]               # (batch_size, 1)
        
        return sigmoid_last, h
    
    
    def init_hidden (self, batch_size):  # initialize hidden weights (h,c) to 0
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
             weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        
        return h


###############################################################################
################  9. INSTANTIATE THE MODEL W/ HYPERPARAMETERS #################
###############################################################################
n_vocab = len(vocab_to_int)
n_embed = 400
n_hidden = 512
n_output = 1   # 1 ("positive") or 0 ("negative")
n_layers = 2

net = SentimentLSTM(n_vocab, n_embed, n_hidden, n_output, n_layers)


###############################################################################
#######################  10. DEFINE LOSS & OPTIMIZER  #########################
###############################################################################
from torch import optim

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)


###############################################################################
##########################  11. TRAIN THE NETWORK!  ###########################
###############################################################################
print_every = 100
step = 0
n_epochs = 4  # validation loss increases from ~ epoch 3 or 4
clip = 5  # for gradient clip to prevent exploding gradient problem in LSTM/RNN
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for epoch in range(n_epochs):
    h = net.init_hidden(batch_size)
    
    for inputs, labels in train_loader:
        step += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        # making requires_grad = False for the latest set of h
        h = tuple([each.data for each in h])   
        
        net.zero_grad()
        output, h = net(inputs)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), clip)
        optimizer.step()
        
        if (step % print_every) == 0:            
            ######################
            ##### VALIDATION #####
            ######################
            net.eval()
            valid_losses = []
            v_h = net.init_hidden(batch_size)
            
            for v_inputs, v_labels in valid_loader:
                v_inputs, v_labels = inputs.to(device), labels.to(device)
        
                v_h = tuple([each.data for each in v_h])
                
                v_output, v_h = net(v_inputs)
                v_loss = criterion(v_output.squeeze(), v_labels.float())
                valid_losses.append(v_loss.item())

            print("Epoch: {}/{}".format((epoch+1), n_epochs),
                  "Step: {}".format(step),
                  "Training Loss: {:.4f}".format(loss.item()),
                  "Validation Loss: {:.4f}".format(np.mean(valid_losses)))
            net.train()


###############################################################################
################  12. TEST THE TRAINED MODEL ON THE TEST SET  #################
###############################################################################
net.eval()
test_losses = []
num_correct = 0
test_h = net.init_hidden(batch_size)

for inputs, labels in test_loader:
    test_h = tuple([each.data for each in test_h])
    test_output, test_h = net(inputs, test_h)
    loss = criterion(test_output, labels)
    test_losses.append(loss.item())
    
    preds = torch.round(test_output.squeeze())
    correct_tensor = preds.eq(labels.float().view_as(preds))
    correct = np.squeeze(correct_tensor.numpy())
    num_correct += np.sum(correct)
    
print("Test Loss: {:.4f}".format(np.mean(test_losses)))
print("Test Accuracy: {:.2f}".format(num_correct/len(test_loader.dataset)))


###############################################################################
############  13. TEST THE TRAINED MODEL ON A RANDOM SINGLE REVIEW ############
###############################################################################
def predict(net, review, seq_length = 200):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    words = preprocess(review)
    encoded_words = [vocab_to_int[word] for word in words]
    padded_words = pad_text([encoded_words], seq_length)
    padded_words = torch.from_numpy(padded_words).to(device)
    
    if(len(padded_words) == 0):
        "Your review must contain at least 1 word!"
        return None
    
    net.eval()
    h = net.init_hidden(1)
    output, h = net(padded_words, h)
    pred = torch.round(output.squeeze())
    msg = "This is a positive review." if pred == 0 else "This is a negative review."
    
    return msg


review1 = "It made me cry."
review2 = "It was so good it made me cry."
review3 = "It's ok."
review4 = "This movie had the best acting and the dialogue was so good. I loved it."
review5 = "Garbage"
                       ### OUTPUT ###
predict(net, review1)  ## negative ##
predict(net, review2)  ## positive ##
predict(net, review3)  ## negative ##
predict(net, review4)  ## positive ##
predict(net, review5)  ## negative ##
    
