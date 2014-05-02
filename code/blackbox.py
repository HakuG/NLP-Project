import Training
import NN
import randomnet


I = 20
H = 40
O = 12
eta = 1
alpha = .4
epochs = 15000

wl = 0
wu = .2
bl = -3
bu = -1
il = 0 
iu = .2
ol = -.5
ou = .2

def main():
    net = NN.NeuralNetwork(I, H, O, eta, alpha, epochs)
    trainingset = randomnet.randomtrainingset(net, wl, wu, bl, bu, il, iu, ol, ou)
    trainednet = Training.train(net, trainingset)
    print "It was supposed to converge to %s\n" % trainingset.outputs
    print "The network is trained!\n"
    # compute outputs
    while True:
    	print "What are your %s inputs?" % trainednet.numInputs
    	inputs = map(float, raw_input("They must be between %s and %s\n" % (il, iu)).split())
    	if len(inputs) != trainednet.numInputs:
    		print "Wrong number of inputs. Start Over\n"
    		break
    	trainednet.inputs = inputs
    	outputs, x, y = Training.ComputeOutputs(trainednet)
    	print "The outputs are %s" % (map(randomnet.trunc,outputs))
   	
main()
