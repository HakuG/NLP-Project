# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 11:35:59 2014

@author: User Name
"""
import math

# define a class of neural networks
class NeuralNetwork:
    
    def __init__(self, numInputs, numHidden, numOutput, eta, alpha):
        self.numInputs = numInputs
        self.numHidden = numHidden
        self.numOutput = numOutput
        self.inputs = [0 for x in xrange (numInputs)]
        self.ihWeights = [[0 for x in xrange(numHidden)] for y in xrange(numInputs)]
        self.hoWeights = [[0 for x in xrange(numOutput)] for y in xrange(numHidden)]
        self.ihSums = [0 for x in xrange(numHidden)]
        self.ihBiases = [0 for x in xrange(numHidden)]        
        self.hoSums = [0 for x in xrange(numOutput)]
        self.hoBiases = [0 for x in xrange(numOutput)]
        self.outputs = [0 for x in xrange(numOutput)]
        self.oGrads = [0 for x in xrange(numOutput)]
        self.hGrads = [0 for x in xrange(numHidden)]
        self.ihWeightsDelta = [[0 for x in xrange(numHidden)] for y in xrange(numInputs)]
        self.ihBiasesDelta = [0 for x in xrange(numHidden)]
        self.hoWeightsDelta = [[0 for x in xrange(numOutput)] for y in xrange(numHidden)]
        self.hoBiasesDelta = [0 for x in xrange(numOutput)]
        self.eta = eta
        self.alpha = alpha        
    
    @staticmethod
    def SigmoidFunction (x):
        if x <-45:
            return 0.0
        elif (x > 45.0):
            return 1.0
        else:
            return 1.0/(1.0 + math.exp(-x))
    @staticmethod
    def HTF(x):
        if x < -10:
            return -1
        elif x > 10:
            return 1
        else:
            return math.tanh(x)
            
def main():

    # print("We will be building a 3 - 4 - 2 neural network.")
    # print("What are the initial 26 random weights and biases?")
    # print("What are the three inputs broski?")
    # print("What is the target output to learn?")
    # print("What is eta and alpha?")

    #initializing everything
    # get number of input nodes
    input_nodes = int(raw_input("How many input nodes?\n"))
    # get number of hidden nodes
    hidden_nodes = int(raw_input("How many hidden nodes?\n"))
    # get number of outpus nodes
    output_nodes = int(raw_input("How many input nodes?\n"))
    # get weights and biases
    wandb = map(float, raw_input("What are the initial 26 random weights and biases?\n").split())
    # get inputs
    inputs = map(float, raw_input("What are the inputs?\n").split())
    # get outputs
    outputs = map(float, raw_input("What are the target outputs to learn?\n").split())
    # get eta
    eta = float(raw_input("What is the eta?\n"))
    # get alpha
    alpha = float(raw_input("What is alpha?\n"))

    net = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, eta, alpha)
    net.inputs = inputs
    net.outputs = outputs
    for i in xrange(net.numInputs):
        for y in xrange(net.numHidden):
            # does this 4 need to be changes?
            net.ihWeights[i][y] = wandb[i * net.numHidden + y]
    for i in xrange(net.numHidden):
        for y in xrange(net.numOutput):
            net.hoWeights[i][y] = wandb[(net.numHidden * net.numInputs) + i * net.numOutput + y]
    for i in xrange(net.numHidden):
        net.ihBiases[i] = wandb[(net.numHidden * net.numInputs + (net.numHidden * net.numOutput)) + i]
    for i in xrange(net.numOutput):
        # prev + hidden
        net.hoBiases[i] = wandb[(net.numHidden * net.numInputs + (net.numHidden * net.numOutput))+ net.numHidden + i]
    for i in xrange(300):
        localout, hlocalout,inputstuff = ComputeOutputs(net)
        ComputeBackPropogation(localout, hlocalout, net)
        ComputeWeightBiasDelta (net, inputstuff)
    ComputeOutputs(net)
def ComputeWeightBiasDelta (net,inputstuff):
    prevdeltih = net.ihWeightsDelta
    prevdeltho = net.hoWeightsDelta  
    prevdelthob = net.hoBiasesDelta
    prevdeltihb = net.ihBiasesDelta
    for i in xrange(net.numInputs):
        for x in xrange(net.numOutput):
            net.ihWeightsDelta[i][x] = net.hGrads[x] * net.eta * net.inputs[i]
            net.ihWeights[i][x] = net.ihWeights[i][x] + net.ihWeightsDelta[i][x] + net.alpha * prevdeltih[i][x]
    for i in xrange(net.numHidden):
        for x in xrange(net.numOutput):
            net.hoWeightsDelta[i][x] = net.oGrads[x] * net.eta * inputstuff[i]
            net.hoWeights [i][x] = net.hoWeights[i][x] + net.hoWeightsDelta[i][x] + net.alpha * prevdeltho[i][x]
    for i in xrange(net.numHidden):
        net.ihBiasesDelta[i] = net.eta * net.hGrads[i]
        net.ihBiases[i] = net.ihBiases[i] + net.ihBiasesDelta[i] + net.alpha * prevdeltihb[i]
    for i in xrange(net.numOutput):
        net.hoBiasesDelta[i] = net.eta * net.oGrads[i]
        net.hoBiases[i] = net.hoBiases[i] + net.hoBiasesDelta[i] + net.alpha * prevdelthob[i]
def ComputeBackPropogation(localout, hlocalout, net):
    for i in xrange(net.numOutput):
        net.oGrads[i] = (1 - localout[i]) * ( 1 + localout[i]) * (net.outputs[i] - localout[i])
    for x in xrange(net.numHidden):
        sumx = 0
        for i in xrange(net.numOutput):
            sumx = sumx + net.oGrads[i] * net.hoWeights[x][i]
        net.hGrads[x] = (hlocalout[x])*(1 - hlocalout[x])*(sumx)
    #Computing Weights
def ComputeOutputs(net):
    local = [0 for x in xrange(net.numHidden)]
    endresult = [0 for x in xrange(net.numOutput)]
    # why is this a 4 and not a three?
    inputstuff = [0 for x in xrange(net.numHidden)]
    for i in xrange(net.numHidden):
        sumi = 0
        for x in xrange(net.numInputs):
            sumi = sumi + net.ihWeights[x][i] * net.inputs[x]
        sumi = sumi + net.ihBiases[i]
        inputstuff[i] = sumi
        local[i] = net.SigmoidFunction(sumi)
    for i in xrange(net.numOutput):
        sumi = 0
        for x in xrange(net.numHidden):
            sumi = sumi + net.hoWeights[x][i]* local[x]
        sumi = sumi + net.hoBiases[i]
        endresult[i] = net.HTF(sumi)
    print(endresult)
    return endresult, local, inputstuff
main()          
        
    
    
        