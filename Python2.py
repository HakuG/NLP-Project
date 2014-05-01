# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 11:35:59 2014

@author: User Name
"""
import math

# define a class of neural networks
class NeuralNetwork:
    
    def __init__(self, numInputs, numHidden, numOutput, eta, alpha):
        # parameters
        self.numInputs = numInputs
        self.numHidden = numHidden
        self.numOutput = numOutput
        self.eta = eta
        self.alpha = alpha
        # initialize empty array for inputs
        self.inputs = [0 for x in xrange (numInputs)]
        # initialize array of empty arrays for input to hidden weights
        self.ihWeights = [[0 for x in xrange(numHidden)] for y in xrange(numInputs)]
        # initialize array of empty arrays for hidden to output weights
        self.hoWeights = [[0 for x in xrange(numOutput)] for y in xrange(numHidden)]
        # initialize an empty array for for sums of hidens
        self.ihSums = [0 for x in xrange(numHidden)]
        # initialize empty array for Biases of hiddens
        self.ihBiases = [0 for x in xrange(numHidden)]
        #initlaize empty array for Sums of outputs        
        self.hoSums = [0 for x in xrange(numOutput)]
        # initialize empty array for Biases of outputs 
        self.hoBiases = [0 for x in xrange(numOutput)]
        # initialize empty array for outputs
        self.outputs = [0 for x in xrange(numOutput)]
        # initialize empty array for gradients of outputs
        self.oGrads = [0 for x in xrange(numOutput)]
        # initialize empty array for gradients of hiddens
        self.hGrads = [0 for x in xrange(numHidden)]
        # initialize array of empty arrays corresponding to change in weights from inputs to hidden
        self.ihWeightsDelta = [[0 for x in xrange(numHidden)] for y in xrange(numInputs)]
        # initialize empty array for change in biases for Hiddens.
        self.ihBiasesDelta = [0 for x in xrange(numHidden)]
        # initialize array of empty arrays for change in weights from hidden to outputs
        self.hoWeightsDelta = [[0 for x in xrange(numOutput)] for y in xrange(numHidden)]
        # initialize empty array for change in biases for Outputs
        self.hoBiasesDelta = [0 for x in xrange(numOutput)]
                
    # Sigmoid function conducts the input to hidden computation
    @staticmethod
    def SigmoidFunction (x):
        if x <-45:
            return 0.0
        elif (x > 45.0):
            return 1.0
        else:
            return 1.0/(1.0 + math.exp(-x))

    # Hyperbolic Tangent Function conducts the hidden to output computation
    @staticmethod
    def HTF(x):
        if x < -10:
            return -1
        elif x > 10:
            return 1
        else:
            return math.tanh(x)
            
def main():
    # get number of input nodes
    input_nodes = int(raw_input("How many input nodes?\n"))
    # get number of hidden nodes
    hidden_nodes = int(raw_input("How many hidden nodes?\n"))
    # get number of outpus nodes
    output_nodes = int(raw_input("How many output nodes?\n"))
    # Compute the number of weights needed
    weights_needed = hidden_nodes * (input_nodes + output_nodes)
    # compute the number of biases needed
    biases_needed = hidden_nodes + output_nodes
    # get weights and biases
    wandb = map(float, raw_input("What are the initial random %s weights %s biases?\n" % (weights_needed, biases_needed)).split())
    # get inputs
    inputs = map(float, raw_input("What are the %s inputs?\n" % input_nodes).split())
    # get outputs
    outputs = map(float, raw_input("What are the %s target outputs to learn?\n" % output_nodes).split())
    # get eta
    eta = float(raw_input("What is the eta?\n"))
    # get alpha
    alpha = float(raw_input("What is alpha?\n"))
    # create Neural Network
    net = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, eta, alpha)
    net.inputs = inputs
    net.outputs = outputs
    #  fill input to hidden array with initial weights
    for i in xrange(net.numInputs):
        for y in xrange(net.numHidden):
            net.ihWeights[i][y] = wandb[i * net.numHidden + y]
    # fill hidden to output array with initial weights
    for i in xrange(net.numHidden):
        for y in xrange(net.numOutput):
            net.hoWeights[i][y] = wandb[(net.numHidden * net.numInputs) + i * net.numOutput + y]
    # fill input to hidden array with initial biases
    for i in xrange(net.numHidden):
        net.ihBiases[i] = wandb[(net.numHidden * net.numInputs + (net.numHidden * net.numOutput)) + i]
    # fill hidden to output array with initial biases
    for i in xrange(net.numOutput):
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
        
    
    
        