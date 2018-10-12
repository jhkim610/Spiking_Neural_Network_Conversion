from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import copy
import argparse
import pdb
import math
import numpy as np

import pickle

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MNIST Training')
parser.add_argument('--batch_size', default = 10, type=int,help = 'batch size')
parser.add_argument('--epoch', default = 30, type = int, help = 'number of iteration')
parser.add_argument('--p', default=99.0, type=float, help='p quantile')
parser.add_argument('--train', default=1, type=int, help='1 if train, 0 if test')
args = parser.parse_args()

batch_size = args.batch_size
use_cuda = torch.cuda.is_available()

i=1
val=99
while True:
    val = args.p*i
    if val - int(val)==0:
	break
    i = i*10
val = int(val)

if use_cuda:
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
    cudnn.benchmark = True
else:
    dtype = torch.FloatTensor
    ltype = torch.cuda.LongTensor

# download and transform train dataset
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=True, transform=transforms.Compose([transforms.ToTensor(),])), batch_size=args.batch_size, shuffle=True)

# download and transform test dataset
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=False, transform=transforms.Compose([transforms.ToTensor(),])), batch_size=args.batch_size*100, shuffle=True)


class CNNClassifier(nn.Module):
    """Custom module for a simple convnet classifier"""
    def __init__(self):
        super(CNNClassifier, self).__init__()
        
	self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
	self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
	self.bn2 = nn.BatchNorm2d(32)
        self.conv3= nn.Conv2d(32,64,kernel_size=5,padding=2)
	self.bn3 = nn.BatchNorm2d(64)
	self.fc1 = nn.Linear(576, 10)
	self.bn4 = nn.BatchNorm1d(10)

    def forward(self, x):
        x0 = self.bn1(self.conv1(x))
	x0_ = F.avg_pool2d(F.relu(x0), 2)
	x1 = self.bn2(self.conv2(x0_))
        x1_ = F.avg_pool2d(F.relu(x1), 2)
	x2 = self.bn3(self.conv3(x1_))
	x2_ = F.avg_pool2d(F.relu(x2), 2)
        x2_ = x2_.view(-1, 576)
        x3 = F.relu(self.bn4(self.fc1(x2_)))
	
        return x3


class Trained_Classifier(nn.Module):
    """Custom module for a simple convnet classifier"""
    def __init__(self):
        super(Trained_Classifier, self).__init__()
        
	self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv3= nn.Conv2d(32,64,kernel_size=5,padding=2)
	self.fc1 = nn.Linear(576, 10)
        # self.fc2 = nn.Linear(50, 10)

    def forward(self, x, layer=-1):
        x0 = F.relu(self.conv1(x))
	x0_ = F.avg_pool2d(x0, 2)
	x1 = F.relu(self.conv2(x0_))
        x1_ = F.avg_pool2d(x1, 2)
	x2 = F.relu(self.conv3(x1_))
	x2_ = F.avg_pool2d(x2, 2)
        x2_ = x2_.view(-1, 576)
        x3 = F.relu(self.fc1(x2_))
        
	if layer==0:
	    return x0
	elif layer==1:
	    return x0_
	elif layer==2:
	    return x1
	elif layer==3: 
	    return x1_
	elif layer==4:
	    return x2
	elif layer==5:
	    return x2_
	else:
	    return x3

# create classifier and optimizer objects
cnn = CNNClassifier()
cnn_bn = Trained_Classifier()

cnn = cnn.type(dtype)
cnn_bn = cnn_bn.type(dtype)

best_model = copy.deepcopy(cnn.state_dict())
best_acc = 0

cri = torch.nn.CrossEntropyLoss()
opt = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)
sch = optim.lr_scheduler.StepLR(opt, 1, gamma=0.8)

loss_history = []
acc_history = []

def train(epoch,cnn=cnn):
    cnn.train() # set model in training mode (need this because of dropout)
    sch.step()

    correct = 0
    # dataset API gives us pythonic batching 
    for batch_id, (data, target) in enumerate(train_loader):
        
	data, target = Variable(data).type(dtype), Variable(target).type(ltype)
        
        # forward pass, calculate loss and backprop!
        opt.zero_grad()
        output = cnn(data)

	loss = cri(output, target)
	loss.backward()
	loss_history.append(loss.data[0])
	opt.step()
        
	prediction = output.data.max(1)[1]
	correct += prediction.eq(target.data).cpu().sum()

	acc = float(correct)/((batch_id+1) * args.batch_size) * 100

        if (batch_id+1) % 100 == 0:
            print('epoch: {}, batch_id: {}, train_loss: {:.2f}, train_acc: {:.2f}'.format(epoch, batch_id+1, loss.data[0], acc))

def test(epoch,cnn=cnn):
    
    global best_acc
    cnn.eval() # set model in inference mode (need this because of dropout)
    test_loss = 0
    correct = 0
    
    for data, target in test_loader:
        data, target = Variable(data).type(dtype), Variable(target).type(ltype)
        
        output = cnn(data)
        test_loss += cri(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    accuracy = 100. * correct / len(test_loader.dataset)
    acc_history.append(accuracy)
    print('\nTest set: Avg loss: {:.4f}, Acc: {}/{} ({:.2f}%), Best Acc: {:.2f}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy, best_acc))
    
    if accuracy > best_acc:
	best_acc = accuracy
	best_model = copy.deepcopy(cnn.state_dict())
	torch.save(best_model,'./weights/cnn_{}.pt'.format(val))

def collect(cnn=cnn_bn, layer=-1):
    cnn.eval()

    activations = np.array([])

    for data, target in train_loader:
	data, target = Variable(data).type(dtype), Variable(target).type(ltype)

	activation = cnn(data, layer)
	activation = activation.data.view(-1)
	nonzero = activation != 0
	activation = activation[nonzero]
	size = activation.size()[0]
	idx = np.random.randint(0, size, size/3)
	activations = np.append(activations, activation.cpu().numpy().view()[idx])
    
    size = len(activations)
    activations = np.sort(activations)
    scaler = activations[int(size*args.p/100)]

    return scaler


if args.train ==1:
    for epoch in range(0, args.epoch):
	print("Epoch %d" % epoch)
	train(epoch)
	test(epoch)

    epoch = 0
    cnn.load_state_dict(torch.load('./weights/cnn_{}.pt'.format(val)))

    print('combine bn layers with convolutional layers')

    for i in range(16):
	W = cnn.conv1.weight.data[i]
	b = cnn.conv1.bias.data[i]

	gamma = cnn.bn1.weight.data[i]
	beta = cnn.bn1.bias.data[i]

	mu = cnn.bn1.running_mean[i]
	s = np.sqrt(cnn.bn1.running_var[i])

	cnn_bn.conv1.weight.data[i] = gamma*W/s
	cnn_bn.conv1.bias.data[i] = gamma*(b-mu)/s + beta

    for i in range(32):
	W = cnn.conv2.weight.data[i]
	b = cnn.conv2.bias.data[i]

	gamma = cnn.bn2.weight.data[i]
	beta = cnn.bn2.bias.data[i]

	mu = cnn.bn2.running_mean[i]
	s = np.sqrt(cnn.bn2.running_var[i])

	cnn_bn.conv2.weight.data[i] = gamma*W/s
	cnn_bn.conv2.bias.data[i] = gamma*(b-mu)/s + beta

    for i in range(64):
	W = cnn.conv3.weight.data[i]
	b = cnn.conv3.bias.data[i]

	gamma = cnn.bn3.weight.data[i]
	beta = cnn.bn3.bias.data[i]

	mu = cnn.bn3.running_mean[i]
	s = np.sqrt(cnn.bn3.running_var[i])

	cnn_bn.conv3.weight.data[i] = gamma*W/s
	cnn_bn.conv3.bias.data[i] = gamma*(b-mu)/s + beta

    for i in range(10):
	W = cnn.fc1.weight.data[i]
	b = cnn.fc1.bias.data[i]

	gamma = cnn.bn4.weight.data[i]
	beta = cnn.bn4.bias.data[i]

	mu = cnn.bn4.running_mean[i]
	s = np.sqrt(cnn.bn4.running_var[i])

	cnn_bn.fc1.weight.data[i] = gamma*W/s
	cnn_bn.fc1.bias.data[i] = gamma*(b-mu)/s + beta


    print('test the combined cnn')
    test(100,cnn_bn)  
	
    torch.save(cnn_bn.state_dict(),'./weights/cnn_bn_{}.pt'.format(val))

cnn_bn.load_state_dict(torch.load('./weights/cnn_bn_{}.pt'.format(val)))


print('find the maximum activation for each layer')
scl_fac=[]
for layer in range(7):
    scaler = collect(cnn_bn,layer)
    print('layer {}: {:.4f}'.format(layer, scaler))
    scl_fac.append(scaler)

lambda_fac = scl_fac
app_fac = []
prev_factor = 1

for i in range(len(scl_fac)):
    scale_factor = scl_fac[i]
    applied_factor = scale_factor/prev_factor
    app_fac.append(applied_factor)
    prev_factor = scale_factor
    print('lambda_factor: {:.4f}, applied_factor: {:.4f}'.format(lambda_fac[i], app_fac[i]))

cnn_bn.conv1.weight.data = cnn_bn.conv1.weight.data/app_fac[0]
cnn_bn.conv1.bias.data = cnn_bn.conv1.bias.data/lambda_fac[0]

cnn_bn.conv2.weight.data = cnn_bn.conv2.weight.data/app_fac[2]
cnn_bn.conv2.bias.data = cnn_bn.conv2.bias.data/lambda_fac[2]

cnn_bn.conv3.weight.data = cnn_bn.conv3.weight.data/app_fac[4]
cnn_bn.conv3.bias.data = cnn_bn.conv3.bias.data/lambda_fac[4]

cnn_bn.fc1.weight.data = cnn_bn.fc1.weight.data/app_fac[6]
cnn_bn.fc1.bias.data = cnn_bn.fc1.bias.data/lambda_fac[6]

np.save('./weights/activations_{}.npy'.format(val),np.array(scl_fac))

print('test the scaled cnn')
test(100, cnn_bn)

