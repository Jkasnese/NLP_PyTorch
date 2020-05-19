import torch
import torchtext
from torchtext.datasets import text_classification
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import datetime
from torch.utils.data.dataset import random_split

NGRAMS = 2
if not os.path.isdir('./.data'):
    os.mkdir('./.data')
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None)
BATCH_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_hidden, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        embedded = F.relu(self.fc(embedded))
        return self.fc2(embedded)


VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUM_HIDDEN = 32
NUN_CLASS = len(train_dataset.get_labels())
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_HIDDEN, NUN_CLASS).to(device)

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)

min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])

def run_loop(OPTIMIZER, LEARNING_RATE, NUM_HIDDEN, LOSS_FUNC, BATCH_SIZE, N_EPOCHS):
    # SummaryWriter
    logdir = "runs/embeddings/notrain/" + "%s_lr%f_nHid%d_%s_bs%d" % (OPTIMIZER, LEARNING_RATE, NUM_HIDDEN, LOSS_FUNC, BATCH_SIZE)
    print ("Log directory: " + logdir)
    writer = SummaryWriter(logdir)
    
    print ("Beginning of training at " + datetime.datetime.now().strftime("%Y_%m_%d_-%H_%M_%S"))
    start_time = time.perf_counter()

    for epoch in range(N_EPOCHS):

        train_loss, train_acc = train_func(sub_train_)
        valid_loss, valid_acc = test(sub_valid_)

        # Training summary
        writer.add_scalar('Training Loss',
                train_loss,
                epoch)
        writer.add_scalar('Training Accuracy',
                train_acc,
                epoch)
        writer.add_scalar('Validation Loss',
                valid_loss,
                epoch)
        writer.add_scalar('Validation Accuracy',    
                valid_acc,
                epoch)

        # Early stop
        # if (prev_valid_loss < valid_loss and prev_valid_acc > valid_acc):
        #     if (better_runs == 4):
        #         break;
        #     else:
        #         better_runs += 1
        # else:
        #     better_runs=0
        # prev_valid_loss = valid_loss
        # prev_valid_acc = valid_acc

    print ("Finished training at " + datetime.datetime.now().strftime("%Y_%m_%d_-%H_%M_%S"))
    total_time = int (time.perf_counter() - start_time)

    # writer.add_scalar('total_time', 
    #                     total_time,
    #                     NUM_EPOCHS+1)

    secs = total_time % 60
    mins = (total_time / 60) % 60
    hours = total_time / 3600
    
    # print ("Total time: %d:%d:%d HH:MM:SS" % (hours, mins, secs))
    # with open(logdir + 'time.txt', 'w') as f:
    #     f.write("Total time: %f:%f:%f HH:MM:SS" % (hours, mins, secs))

    # Test
    test_loss, test_acc = test(test_dataset)

    # Parameters summary
    writer.add_hparams({"batch_size":BATCH_SIZE,
                        "num_hidden":NUM_HIDDEN,
                        "optimizer":OPTIMIZER, 
                        "learning_rate":LEARNING_RATE,
                        "loss_function":LOSS_FUNC},
                        {"Test_loss":test_loss, "Test_acc":test_acc, "total_time":total_time})

# Training hyperparameters to be compared
NUM_EPOCHS = 30

H_BATCH_SIZE = [512, 256] # opts , 64, 32, 8
H_NUM_HIDDEN = [256, 64] # opts 128,
H_OPTIMIZER = ["sgd"] # opts , "adam"
H_LEARNING_RATE = [0.1, 0.001] # opts  0.01, 
H_LOSS_FUNC = ["cross"] # opts , "nll"

run_loop("sgd", 0.1, 32, "cross", 256, 30)