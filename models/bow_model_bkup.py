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
"""

BATCH_SIZE = [8, 32] # opts
NUM_HIDDEN = [100, 300] # opts 
OPTIMIZER = ["sgd", "adam"] # opts 
LEARNING_RATE = [0.1, 0.001] # opts 
LOSS_FUNC = ["cross", "nll"] # opts
"""

# Load data and make batches
print ("Reading data")
vocabulary, data = load_vocab_data()
print("Vocabulary len is: %d. Divided by 8 is %d" % (len(vocabulary)), (len(vocabulary) % 8))
train_len = int (len (data) * 0.9)
sub_train, sub_valid = random_split(data, [train_len, (len(data) - train_len)] )
print ("Data read and split!")

def generate_batch(data):
    label = torch.LongTensor([(int(entry[1])-1) for entry in data])
    sentence = torch.Tensor([make_bow_vector(entry[0], vocabulary) for entry in data])
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

def train(sub_train_, model, batch_size, loss_function, optimizer, writer):
    train_loss = 0
    train_acc = 0

    data_iter = DataLoader(sub_train_, batch_size=batch_size, shuffle=True, collate_fn=generate_batch, pin_memory=True)
    
    for i, (sentences, label) in enumerate(data_iter):
        train_loss = 0
        train_acc = 0
        sentences, label = sentences.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(sentences)
        loss = loss_function(output, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        train_acc += (output.argmax(1) == label).sum().item()

        writer.add_scalar('Training Loss',
                        train_loss,
                        (i+1))
        writer.add_scalar('Training Accuracy',
                        train_acc/batch_size,
                        (i+1))

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(sub_test_, model, batch_size, loss_function):
    test_acc = 0
    test_loss = 0

    data_iter = DataLoader(sub_test_, batch_size=batch_size, collate_fn=generate_batch, pin_memory=True)

    for sentences, label in data_iter:
        sentences, label = sentences.to(device), label.to(device)
        output = model(sentences)
        loss = loss_function(output, label)
        test_loss += loss.item()
        test_acc += (output.argmax(1) == label).sum().item()
    
    return test_acc / len(sub_test_), test_loss / len(sub_test_)

def training_loop(BATCH_SIZE, NUM_HIDDEN, OPTIMIZER, LEARNING_RATE, LOSS_FUNC):

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
    data_iter = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch).__iter__()
    input_, _ = next(data_iter)
    writer.add_graph(model, input_)
    writer.close()
    model = model.to(device)
    print ("Model generated and sent to device")

    if (LOSS_FUNC == "cross"):
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.NLLLoss()
    if (OPTIMIZER == "sgd"):
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    valid_acc =0.0
    prev_valid_loss = float('inf')

    print ("Beginning of training at " + datetime.datetime.now().strftime("%Y_%m_%d_-%H_%M_%S"))
    start_time = time.perf_counter()

    for epoch in range(1):
        train_acc = 0.0
        train_loss = 0.0

        previous_time = time.perf_counter()

        train_acc, train_loss = train(sub_train, model, BATCH_SIZE, loss_function, optimizer, writer)
        valid_acc, valid_loss = test(sub_valid, model, BATCH_SIZE, loss_function)

        # if (prev_valid_loss - valid_loss < )

        # print statistics
        # writer.add_scalar('training_loss',
        #                     train_loss,
        #                     (epoch+1))
        # writer.add_scalar('validation_loss',
        #                     valid_loss,
        #                     (epoch+1))
        # print('[%d] Train loss: %.3f - Train acc: %.2f' %
        #     (epoch + 1, train_loss, train_acc))
        # print('[%d] Valid loss: %.3f - Valid acc: %.2f' %
        #     (epoch + 1, valid_loss, valid_acc))
        # print ('Epoch time: %.2f' % (time.perf_counter() - previous_time))

    total_time = int (time.perf_counter() - start_time)

    writer.add_hparams({"batch_size":BATCH_SIZE,
                        "num_hidden":NUM_HIDDEN,
                        "optimizer":OPTIMIZER, 
                        "learning_rate":LEARNING_RATE,
                        "loss_function":LOSS_FUNC},
                        {"Test_acc":valid_acc, "total_time":total_time})

    writer.add_scalar('total_time', 
                        total_time,
                        len(data)+1)

    secs = total_time % 60
    mins = (total_time / 60) % 60
    hours = total_time / 3600

    print ("Finished training at " + datetime.datetime.now().strftime("%Y_%m_%d_-%H_%M_%S"))
    print ("Total time: %d:%d:%d HH:MM:SS" % (hours, mins, secs))
    with open(logdir + 'time.txt', 'w') as f:
        f.write("Test acc: %.2f" % valid_acc)
        f.write(("Total time: %d:%d:%d HH:MM:SS" % (hours, mins, secs)))
        fwrite("Parameters: " +
                        "\nbatch_size: " + str(BATCH_SIZE) +
                        "\nnum_hidden: " + str(NUM_HIDDEN) +
                        "\noptimizer " + str(OPTIMIZER) +
                        "\nlearning_rate " + str(LEARNING_RATE) +
                        "\nloss_function " + str(LOSS_FUNC))

H_BATCH_SIZE = [8, 32]
H_NUM_HIDDEN = [100, 300, 500]
H_OPTIMIZER = ["sgd", "adam"]
H_LEARNING_RATE = [0.1, 0.01, 0.001]
H_LOSS_FUNC = ["cross", "nll"]

# for i in H_BATCH_SIZE:
#     for j in H_NUM_HIDDEN:
#         for k in H_OPTIMIZER:
#             for l in H_LEARNING_RATE:
#                 for m in H_LOSS_FUNC:
#                     training_loop(i, j, k, l, m)

sgd = "sgd"
cross="cross"
training_loop(32, 300, "sgd", 0.1, "cross")
cProfile.run("training_loop(32, 300, sgd, 0.1, cross)", "training_profile")



# test_data = load_test_data()

# Test
# with torch.no_grad():
#    for instance, label in test_data:
#        bow_vec = make_bow_vector(instance, vocabulary)
#        log_probs = model(bow_vec)
#    print(log_probs)
