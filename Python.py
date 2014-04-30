# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 11:35:59 2014

@author: User Name
"""
import math

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
    #initializing everything
    print("We will be building a 3 - 4 - 2 neural network.")
    print("What are the initial 26 random weights and biases?")
    print("What are the three inputs broski?")
    print("What is the target output to learn?")
    print("What is eta and alpha?")
    wandb = map(float, raw_input().split())
    inputs = map(float, raw_input().split())
    outputs = map(float, raw_input().split())
    eta, alpha = map(float, raw_input().split())
    net = NeuralNetwork(3, 4, 2, eta, alpha)
    net.inputs = inputs
    net.outputs = outputs
    for i in xrange(3):
        for y in xrange(4):
            net.ihWeights[i][y] = wandb[i * 4 + y]
    for i in xrange(4):
        for y in xrange(2):
            net.hoWeights[i][y] = wandb[12 + i * 2 + y]
    for i in xrange(4):
        net.ihBiases[i] = wandb[20 + i]
    for i in xrange(2):
        net.hoBiases[i] = wandb[24 + i]
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
    for i in xrange(3):
        for x in xrange(4):
            net.ihWeightsDelta[i][x] = net.hGrads[x] * net.eta * net.inputs[i]
            net.ihWeights[i][x] = net.ihWeights[i][x] + net.ihWeightsDelta[i][x] + net.alpha * prevdeltih[i][x]
    for i in xrange(4):
        for x in xrange(2):
            net.hoWeightsDelta[i][x] = net.oGrads[x] * net.eta * inputstuff[i]
            net.hoWeights [i][x] = net.hoWeights[i][x] + net.hoWeightsDelta[i][x] + net.alpha * prevdeltho[i][x]
    for i in xrange(4):
        net.ihBiasesDelta[i] = net.eta * net.hGrads[i]
        net.ihBiases[i] = net.ihBiases[i] + net.ihBiasesDelta[i] + net.alpha * prevdeltihb[i]
    for i in xrange(2):
        net.hoBiasesDelta[i] = net.eta * net.oGrads[i]
        net.hoBiases[i] = net.hoBiases[i] + net.hoBiasesDelta[i] + net.alpha * prevdelthob[i]
def ComputeBackPropogation(localout, hlocalout, net):
    for i in xrange(2):
        net.oGrads[i] = (1 - localout[i]) * ( 1 + localout[i]) * (net.outputs[i] - localout[i])
    for x in xrange(4):
        sumx = 0
        for i in xrange(2):
            sumx = sumx + net.oGrads[i] * net.hoWeights[x][i]
        net.hGrads[x] = (hlocalout[x])*(1 - hlocalout[x])*(sumx)
    #Computing Weights
def ComputeOutputs(net):
    local = [0 for x in xrange(4)]
    endresult = [0 for x in xrange(2)]
    inputstuff = [0 for x in xrange(4)]
    for i in xrange(4):
        sumi = 0
        for x in xrange(3):
            sumi = sumi + net.ihWeights[x][i] * net.inputs[x]
        sumi = sumi + net.ihBiases[i]
        inputstuff[i] = sumi
        local[i] = net.SigmoidFunction(sumi)
    for i in xrange(2):
        sumi = 0
        for x in xrange(4):
            sumi = sumi + net.hoWeights[x][i]* local[x]
        sumi = sumi + net.hoBiases[i]
        endresult[i] = net.HTF(sumi)
    print(endresult)
    return endresult, local, inputstuff
main()          
        
    
    
        