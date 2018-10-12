from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np

import copy
import argparse
import pdb
import math

import pickle

import matplotlib.pyplot as plt

from termcolor import colored, cprint


import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MNIST Training')
parser.add_argument('--batch_size', default = 100, type=int,help = 'batch size')
parser.add_argument('--time_step', default = 100, type=int, help='time step')
parser.add_argument('--Vth', default=1.0,type=float, help='threshold voltage')
parser.add_argument('--ttf', default=0, type=int, help='0 if rate coding, 1 if ttf coding')
parser.add_argument('--p',default=99,type=int, help='p quantile')
args = parser.parse_args()

# Vth = args.Vth
batch_size = args.batch_size
time_step = args.time_step
use_cuda = torch.cuda.is_available()
ttf = args.ttf
times = []

coding = "rate"
if ttf==1:
    coding="time to first"

cprint("spiking neural network",'blue')
print("coding: "+coding)
if ttf==0:
    print("time step: {}".format(time_step))
print("batch size: {}".format(batch_size))

if use_cuda:
    dtype = torch.cuda.FloatTensor
    cudnn.benchmark = True
else:
    dtype = torch.FloatTensor


def dataloader():
    # download and transform train dataset
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=True, transform=transforms.Compose([transforms.ToTensor(),])),batch_size=args.batch_size,shuffle=True)

    # download and transform test dataset
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=False, transform=transforms.Compose([transforms.ToTensor(),])),batch_size=batch_size,shuffle=True)

    return train_loader, test_loader

class SNNClassifier(nn.Module):
    """Custom module for a simple convnet classifier"""
    def __init__(self,Vth,batch_size=batch_size):
        super(SNNClassifier, self).__init__()
	self.batch_size=batch_size
	self.Vth = Vth
	self.conv1 = nn.Conv2d(1, 16, kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5,padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5,padding=2)
	self.fc1 = nn.Linear(576, 10)


    def forward(self, x, Vth=np.ones([7])):

	conv1_mem = Variable(torch.zeros((self.batch_size, 16,28,28))).type(dtype)
	pool1_mem = Variable(torch.zeros((self.batch_size, 16,14,14))).type(dtype)
	

	l2_bm = Variable(torch.zeros((self.batch_size))).type(dtype)
	conv2_mem = Variable(torch.zeros((self.batch_size, 32,14,14))).type(dtype)
	pool2_mem = Variable(torch.zeros((self.batch_size, 32,7,7))).type(dtype)
	
	l3_bm = Variable(torch.zeros((self.batch_size))).type(dtype)
	conv3_mem = Variable(torch.zeros((self.batch_size, 64,7,7))).type(dtype)
	pool3_mem = Variable(torch.zeros((self.batch_size, 64,3,3))).type(dtype)
	

	l4_bm = Variable(torch.zeros((self.batch_size))).type(dtype)
	fc1_mem = Variable(torch.zeros((self.batch_size, 10)).type(dtype))
	output = Variable(torch.zeros((self.batch_size,10)).type(dtype))
	returns = Variable(torch.zeros((self.batch_size,10)).type(dtype))
	checked = np.zeros([self.batch_size])

	for t in range(time_step):
	    # spike_train = x/scaling > Variable(torch.rand(x.size()).type(dtype))
	    conv1_mem += self.conv1(x)
	    spike = conv1_mem >= self.Vth[0]
	    conv1_mem[spike.data] -= self.Vth[0]

	    pool1_mem += F.avg_pool2d(spike.type(dtype),2)
	    spike = pool1_mem>=self.Vth[1]
	    pool1_mem[spike.data] -= self.Vth[1]

	    l2_bm += spike.sum(3).sum(2).sum(1).type(dtype)
	    z2 = self.conv2(spike.type(dtype))
	    for i in range(self.batch_size):
		if l2_bm[i].data.cpu().numpy() == 0:
		    z2[i] = z2[i]*0
	    conv2_mem += z2
	    spike = conv2_mem >= self.Vth[2]
	    conv2_mem[spike.data] -= self.Vth[2]

	    pool2_mem += F.avg_pool2d(spike.type(dtype),2)
	    spike = pool2_mem>=self.Vth[3]
	    pool2_mem[spike.data] -= self.Vth[3]

	    l3_bm += spike.sum(3).sum(2).sum(1).type(dtype)
	    z3 = self.conv3(spike.type(dtype))    
	    for i in range(self.batch_size):
		if l3_bm[i].data.cpu().numpy() == 0:
		    z3[i] = z3[i]*0
	    conv3_mem += z3
	    spike = conv3_mem >= self.Vth[4]
	    conv3_mem[spike.data] -= self.Vth[4]

	    pool3_mem += F.avg_pool2d(spike.type(dtype),2)
	    spike = pool3_mem>=self.Vth[5]
	    pool3_mem[spike.data] -= self.Vth[5]

	    spike = spike.view(-1,576)

	    l4_bm += spike.sum(1).type(dtype)
	    z4 = self.fc1(spike.type(dtype))
	    for i in range(self.batch_size):
		if l4_bm[i].data.cpu().numpy() == 0:
		    z4[i] = z4[i]*0
	    fc1_mem += z4
	    spike = fc1_mem >= self.Vth[6]
	    fc1_mem[spike.data] -= self.Vth[6]
	    output += spike.type(dtype)
	    
	    if ttf==1:
		for i in range(self.batch_size):
		    if (output[i].sum().data.cpu().numpy()>0)[0] and checked[i]==0:
			returns[i] = output[i]
			checked[i] = 1
			times.append(t)
	    else:
		returns = output
	    
	    complete=1
	    for i in range(len(checked)):
		complete = complete*checked[i]
	    if complete==1:
		break

	return returns

def test(clf):
    _, loader = dataloader()


    cri = torch.nn.CrossEntropyLoss()

   
    clf.eval() # set model in inference mode (need this because of dropout)
    test_loss = 0
    correct = 0
 
    k = 0
    for data, target in loader:
	k += 1
	if use_cuda:
	    data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)
	
        output = clf(data)
        test_loss += cri(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

	accuracy = 100. * correct / (batch_size*k)
	cprint('Test set: Acc: {}/{} ({:.2f}%)'.format(correct, k*batch_size,accuracy),'yellow')



if __name__=="__main__":
	
    cprint('weight initialization with normalization','blue')
    ## weight normalization factors
    scl_fac = np.load('./weights/activations_{}.npy'.format(args.p))

    lambda_fac = scl_fac

    prev_factor = 1
    app_fac = []
    for i in range(len(scl_fac)):
	scale_factor = scl_fac[i]
	applied_factor = scale_factor/prev_factor
	prev_factor = scale_factor
	print('lambda_factor: {:.4f}, applied_factor(Vth) {:.4f}'.format(lambda_fac[i],applied_factor))
	app_fac.append(applied_factor)
    

    # create classifier and optimizer objects

    clf = SNNClassifier(Vth=app_fac)
    clf = clf.type(dtype)
    clf.load_state_dict(torch.load('./weights/cnn_bn_{}.pt'.format(args.p)))

    clf.conv1.bias.data = clf.conv1.bias.data/lambda_fac[0]*app_fac[0]
    clf.conv2.bias.data = clf.conv2.bias.data/lambda_fac[2]*app_fac[2]
    clf.conv3.bias.data = clf.conv3.bias.data/lambda_fac[4]*app_fac[4]
    clf.fc1.bias.data = clf.fc1.bias.data /lambda_fac[6]*app_fac[6]

    cprint('Test','red')
    test(clf)

    if ttf==1:
	times = np.array(times)

	avg_time = np.mean(times)
	cprint("average inference time is {:.4f}".format(avg_time))
