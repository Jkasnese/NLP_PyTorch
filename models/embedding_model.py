import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torchtext.datasets import text_classification
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PERSIST = False

NGRAMS = 2
if not os.path.isdir('./.data'):
    os.mkdir('./.data')
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None)
VOCAB_SIZE = len(train_dataset.get_vocab())
NUN_CLASS = len(train_dataset.get_labels())
train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])

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
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc2.bias.data.zero_()

    def forward(self, text, offsets):
        out = self.embedding(text, offsets)
        out = F.relu(self.fc(out))
        out = self.fc2(out)
        return out

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

def train_func(sub_train_, model, batch_size, criterion, optimizer, scheduler):
    model.train()

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=batch_size, shuffle=True,
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

def test(data_, model, batch_size, criterion):
    model.eval()
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=batch_size, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)

def run_loop(OPTIMIZER, LEARNING_RATE, NUM_HIDDEN, EMBED_DIM, LOSS_FUNC, BATCH_SIZE, N_EPOCHS, num_run):
    # Setup run
    model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_HIDDEN, NUN_CLASS).to(device)
    if (LOSS_FUNC == "cross"):
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.NLLLoss().to(device)

    if (OPTIMIZER == "sgd"):
        optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=4.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        
    min_valid_loss = float('inf')
    prev_valid_loss = float('inf')
    prev_valid_acc = 0.0
    worst_runs = 0

    # Setup SummaryWriter
    logdir = "runs/exps/nHidden_EmbeddedDim/" + "%s_lr%f_nHid%d_nEmb%d_%s_bs%d/" \
            % (OPTIMIZER, LEARNING_RATE, NUM_HIDDEN, EMBED_DIM, LOSS_FUNC, BATCH_SIZE) # runs/embeddings/notrain/
    rundir = logdir + "run_%d" % (num_run)
    print ("Log directory: " + rundir)
    writer = SummaryWriter(rundir)
    
    print ("Beginning of training at " + datetime.datetime.now().strftime("%Y_%m_%d_-%H_%M_%S"))
    start_time = time.perf_counter()

    # Training loop
    for epoch in range(N_EPOCHS):

        train_loss, train_acc = train_func(sub_train_, model, BATCH_SIZE, criterion, optimizer, scheduler)
        valid_loss, valid_acc = test(sub_valid_, model, BATCH_SIZE, criterion)

        # Training summary
        writer.add_scalar('Training Loss',
                train_loss,
                epoch+1)
        writer.add_scalar('Training Accuracy',
                train_acc,
                epoch+1)
        writer.add_scalar('Validation Loss',
                valid_loss,
                epoch+1)
        writer.add_scalar('Validation Accuracy',    
                valid_acc,
                epoch+1)

        # Model saving
        if (valid_loss < min_valid_loss):
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()
            }, rundir + '.tmp/model.tar')

        # Early stop
        if (prev_valid_loss < valid_loss and prev_valid_acc > valid_acc):
            # If 4 worst runs in sequence, break
            if (worst_runs == 4):
                break;
            else:
                worst_runs += 1
        else:
            worst_runs=0
        prev_valid_loss = valid_loss
        prev_valid_acc = valid_acc


    print ("Finished training at " + datetime.datetime.now().strftime("%Y_%m_%d_-%H_%M_%S"))
    total_time = int (time.perf_counter() - start_time)

    # Test
    checkpoint = torch.load(rundir + '.tmp/model.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = test(test_dataset, model, BATCH_SIZE, criterion)

    # Parameters summary
    writer.add_hparams({"optimizer":OPTIMIZER, 
                        "learning_rate":LEARNING_RATE,
                        "num_hidden":NUM_HIDDEN,
                        "Embedded_Dimensions":EMBED_DIM,
                        "loss_function":LOSS_FUNC,
                        "batch_size":BATCH_SIZE,
                        "num_Epochs":N_EPOCHS},
                        {"Test_loss":test_loss, "Test_acc":test_acc, "Best_Epoch":checkpoint['epoch'], "total_time":total_time})

    # Write experiment results, timings and model
    with open (logdir + 'results.csv', 'a') as f:
        f.write(f'{test_loss}, {test_acc}\n')

    with open (logdir + 'timings.txt', 'a') as f:
        f.write(f'{total_time}\n')

    if (MODEL_PERSIST):
        with open (rundir + 'model.tar', 'w') as f:
            f.write(checkpoint)

# Training hyperparameters to be compared
H_OPTIMIZER = ["sgd"] # opts , "adam"
H_LEARNING_RATE = [4.0] # opts  0.01, 
H_NUM_HIDDEN = [2048, 512, 128, 32] # opts 128,
H_EMBED_DIM = [512, 256, 32] # , 64, 128, 256
H_LOSS_FUNC = ["cross"] # opts , "nll"
H_BATCH_SIZE = [512] # opts 256, 64, 32, 8
H_NUM_EPOCHS = [100]

NUM_RUNS = 5 # Number of runs to calculate mean and sd from

# Setup experiment for each hyperparameters
for opt in H_OPTIMIZER:
    for lr in H_LEARNING_RATE:
        for nHid in H_NUM_HIDDEN:
            for emb in H_EMBED_DIM:
                for lf in H_LOSS_FUNC:
                    for bs in H_BATCH_SIZE:
                        for nEpoc in H_NUM_EPOCHS:
                            for r in range(NUM_RUNS):
                                run_loop(opt, lr, nHid, emb, lf, bs, nEpoc, r)

# run_loop("sgd", 0.1, 32, 32, "cross", 256, 100)