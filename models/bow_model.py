# TODO
# Separate files according to responsability on the code

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import time
import datetime
import sys
import cProfile
sys.path.append('../embeddings')

from one_hot_vec import *

# Torch conf
if torch.cuda.is_available():
    device=torch.device("cuda")
    print ("Device: cuda")
else:
    device=torch.device("cpu")
    print ("Device: cpu")

# Training variables and hyperparameters
NUM_LABELS = 4

H_BATCH_SIZE = [256, 64, 32, 8] # opts
H_NUM_HIDDEN = [256, 128, 64] # opts
H_OPTIMIZER = ["sgd", "adam"] # opts
H_LEARNING_RATE = [0.1, 0.01, 0.001] # opts
H_LOSS_FUNC = ["cross", "nll"] # opts

# Load data and make batches
print ("Reading data")
vocabulary, data = load_vocab_data()
print("Vocabulary len is: %d." % (len(vocabulary)))
data_len = len(data)
train_len = int (data_len * 0.9)
sub_train, sub_valid = random_split(data, [train_len, (len(data) - train_len)] )
print ("Data read and split!")

# TODO: 
# Open issue about for loop.
# Consider transforming generate_batch into a generator?
def generate_batch(data):
    for entry in data:
        sentence = [make_bow_vector(entry[0], vocabulary)]
        label = [int(entry[1]) - 1]
    sentence = torch.FloatTensor(sentence)
    label = torch.tensor(label)
    return sentence, label

class BowModel(nn.Module):
    def __init__(self, num_class, vocab_size, num_hidden):
        super(BowModel, self).__init__()
        # The shape of the linear layer is [vocab_size, num_class]
        self.wb1 = nn.Linear(vocab_size, num_hidden)
        self.wb2 = nn.Linear(num_hidden, num_class)

    def forward(self, bow_vec):
        bow_vec = F.relu(self.wb1(bow_vec))
        bow_vec = self.wb2(bow_vec)
        return F.log_softmax(bow_vec, dim=1)

def train(sentences, label, model, batch_size, loss_function, optimizer):
    train_loss = 0
    train_acc = 0

    optimizer.zero_grad()
    output = model(sentences)
    loss = loss_function(output, label)
    train_loss += loss.item()
    loss.backward()
    optimizer.step()

    train_acc += (output.argmax(1) == label).sum().item()

    return train_loss, train_acc

def test(sub_test_, model, batch_size, loss_function):
    test_acc = 0
    test_loss = 0
    
    with torch.no_grad():
        data_iter = DataLoader(sub_test_, batch_size=batch_size, collate_fn=generate_batch, pin_memory=True)

        for sentences, label in data_iter:
            sentences, label = sentences.to(device), label.to(device)
            output = model(sentences)
            loss = loss_function(output, label)
            test_loss += loss.item()
            test_acc += (output.argmax(1) == label).sum().item()

    return test_loss / len(sub_test_), test_acc / len(sub_test_)

def run_loop(BATCH_SIZE, NUM_HIDDEN, OPTIMIZER, LEARNING_RATE, LOSS_FUNC):

    logdir = "runs/" + "%s_lr%f_nHid%d_%s_bs%d" % (OPTIMIZER, LEARNING_RATE, NUM_HIDDEN, LOSS_FUNC, BATCH_SIZE)
    print ("Log directory: " + logdir)
    writer = SummaryWriter(logdir)
  
    """
    Parameters:
    BATCH_SIZE = 32
    NUM_HIDDEN = 300
    OPTIMIZER = "sgd"
    LEARNING_RATE = 0.1
    LOSS_FUNC = "cross"
    """

    # Create model and add to Tensorboard
    model = BowModel(NUM_LABELS, len(vocabulary), NUM_HIDDEN)
    model = model.to(device)
    data_iter = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch).__iter__()
    input_, _ = next(data_iter)
    input_ = input_.to(device)
    writer.add_graph(model, input_)
    writer.close()
    # print ("Model generated and sent to device")

    if (LOSS_FUNC == "cross"):
        loss_function = nn.CrossEntropyLoss().to(device)
    else:
        loss_function = nn.NLLLoss().to(device)
    if (OPTIMIZER == "sgd"):
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    valid_acc = 0.0
    prev_valid_loss = float('inf')
    prev_valid_acc = 0.0
    better_runs = 0

    # Decide the granularity of the plot
    min_batch_size = min(H_BATCH_SIZE)
    # samples_to_info = int(data_len*0.01) # Percentage of samples before showing info
    # samples_to_info = samples_to_info / max(H_BATCH_SIZE) # How many steps the max batch_size have to take to show info

    print ("Beginning of training at " + datetime.datetime.now().strftime("%Y_%m_%d_-%H_%M_%S"))
    start_time = time.perf_counter()

    for epoch in range(1):
        train_acc = 0.0
        train_loss = 0.0
        cum_train_loss = 0.0
        cum_train_acc = 0.0

        # previous_time = time.perf_counter()

        data_iter = DataLoader(sub_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch, pin_memory=True)

        for i, (sentences, label) in enumerate(data_iter):
            sentences, label = sentences.to(device), label.to(device)
            train_loss, train_acc = train(sentences, label, model, BATCH_SIZE, loss_function, optimizer)
            cum_train_loss += train_loss
            cum_train_acc += train_acc

            # if (i * BATCH_SIZE % samples_to_info == 0): # Granularity of plot
            # valid_loss, valid_acc = test(sub_valid, model, BATCH_SIZE, loss_function)
            writer.add_scalar('Training Loss',
                (cum_train_loss / (i+1)*BATCH_SIZE),
                (i+1)*BATCH_SIZE*(epoch+1)/min_batch_size) # Number of samples normalized by 
            writer.add_scalar('Training Accuracy',
                (cum_train_acc / (i+1)*BATCH_SIZE),
                (i+1)*BATCH_SIZE*(epoch+1)/min_batch_size)
            writer.add_scalar('Validation Loss',
                train_loss,
                (i+1)*BATCH_SIZE*(epoch+1)/min_batch_size)
            writer.add_scalar('Validation Accuracy',    
                train_acc,
                (i+1)*BATCH_SIZE*(epoch+1)/min_batch_size)
                
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

    # Beginning of testing
    test_data = load_test_data()
    print ("Len Test Data: %d" % len(test_data))
    test_acc, test_loss = test(test_data, model, BATCH_SIZE, loss_function)

    total_time = int (time.perf_counter() - start_time)

    writer.add_hparams({"batch_size":BATCH_SIZE,
                        "num_hidden":NUM_HIDDEN,
                        "optimizer":OPTIMIZER, 
                        "learning_rate":LEARNING_RATE,
                        "loss_function":LOSS_FUNC},
                        {"Test_acc":test_acc, "Test_loss":test_loss, "total_time":total_time})

    writer.add_scalar('total_time', 
                        total_time,
                        0)

    secs = total_time % 60
    mins = (total_time / 60) % 60
    hours = total_time / 3600
    
    print ("Finished training at " + datetime.datetime.now().strftime("%Y_%m_%d_-%H_%M_%S"))
    print ("Total time: %d:%d:%d HH:MM:SS" % (hours, mins, secs))
    with open(logdir + 'time.txt', 'w') as f:
        f.write("Test acc: %.2f" % valid_acc)
        f.write("Total time: %f:%f:%f HH:MM:SS" % (hours, mins, secs))
        f.write("Parameters: " +
                        "\nbatch_size: " + str(BATCH_SIZE) +
                        "\nnum_hidden: " + str(NUM_HIDDEN) +
                        "\noptimizer " + str(OPTIMIZER) +
                        "\nlearning_rate " + str(LEARNING_RATE) +
                        "\nloss_function " + str(LOSS_FUNC))

# for i in H_BATCH_SIZE:
#     for j in H_NUM_HIDDEN:
#         for k in H_OPTIMIZER:
#             for l in H_LEARNING_RATE:
#                 for m in H_LOSS_FUNC:
#                     run_loop(i, j, k, l, m)

# run_loop(256, 64, "sgd", 0.1, "cross")



sgd = "sgd"
cross="cross"
cProfile.run("run_loop(256, 64, sgd, 0.1, cross)", "valid_commented.cProfile.prof")

with open ("gen_batch.autograd.prof", 'w') as f:
    f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))