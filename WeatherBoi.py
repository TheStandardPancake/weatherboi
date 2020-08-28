#This code was written entirely by me, Boyd Kirkman - STARTED: 11/7/2020

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#IMPORTS

import numpy as np
import csv

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#MATH FUNCTIONS

#the normalising sigmoid function
def SigmoidFreud(x):
    return 1/(1+np.exp(-x))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#VARIABLE ASSIGNING

#setting seed for predictable "random" generation
np.random.seed(69)
#inputs
_correctAnswer = np.array([])
_inputs = np.array([])
#Connections to hidden layer
_weights1 = np.random.rand(4,5)
_weights1Changes = np.array([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]])
_bias1 = np.random.rand(4)
_bias1Change = np.array([[],[],[],[]])
_hiddenLayer1 = []
#Connections to output layer
_weights2 = np.random.rand(4)
_weights2Changes = np.array([[],[],[],[]])
_bias2 = np.random.rand(1)
_bias2Change = np.array([])

#defining the number of cycles the neural net will train for, as well as the learning rate
_trainingCycles = 1
_learningRate = 1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#CONSTRUCTING THE NUERAL NETWORK

#making a class that will make neuron creation, calculation and organisation easier.
class neuron():
    def __init__(self, weights, inputs, bias):
        self.weights = weights
        self.inputs = inputs
        self.bias = bias
        self.value = 0

    def calcSelf(self): #calculate the neuron's value between 1 and 0
        self.value = SigmoidFreud(np.sum(np.multiply(self.weights,self.inputs))+self.bias)
        return self.value

#training the neurons:
def train():
    global _weights1
    global _bias1
    global _weights2
    global _bias2
    global neuron
    global _learningRate
    global _correctAnswer
    print(_inputs)
    for cycles in range(_trainingCycles):
        print(f"Cycle: {cycles}")
        neuron1 = neuron(_weights1[0],_inputs[cycles],_bias1[0])
        neuron2 = neuron(_weights1[1],_inputs[cycles],_bias1[1])
        neuron3 = neuron(_weights1[2],_inputs[cycles],_bias1[2])
        neuron4 = neuron(_weights1[3],_inputs[cycles],_bias1[3])
        _hiddenLayer1 = np.array([neuron1.calcSelf(),neuron2.calcSelf(),neuron3.calcSelf(),neuron4.calcSelf()])
        _output = SigmoidFreud(np.sum(np.multiply(_weights2,_hiddenLayer1))+_bias2) #calculation of the output neuron and normalising to produce a number between 0 and 1
        print(f"output: {_output[0]}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#BACKPROPIGATION OF THE AI --- NOTE This is still inside the train() function

#1> Calculating the required weight and bias changes
        _memoization = -_learningRate*(_output[0]*(1-_output[0]))*2(_output[0]-_correctAnswer[cycles]) #reduces the number of times the computer calculates this value
        for neuron in range(4):
            _w2Change = 0
            _b1Change = 0
            if neuron == 0:
                _w2Change = neuron1.value*_memoization
                _b1Change = -_learningRate*(neuron1.value*(1-neuron1.value))*2(_output[0]-_correctAnswer[cycles])
                _bias1Change[0].append(_b1Change) #placing these values into a list to average them and make a change later
                for w in range(5):
                    _w1Change = neuron1.inputs[w]*_memoization
                    _weights1Changes[w].append(_w1Change) #placing these values into a list to average them and make a change later
            if neuron == 1:
                _w2Change = neuron2.value*_memoization
                _b1Change = -_learningRate*(neuron2.value*(1-neuron2.value))*2(_output[0]-_correctAnswer[cycles])
                _bias1Change[1].append(_b1Change)
                for w in range(5):
                    _w1Change = neuron2.inputs[w]*_memoization
                    _weights1Changes[w+5].append(_w1Change)
            if neuron == 2:
                _w2Change = neuron3.value*_memoization
                _b1Change = -_learningRate*(neuron3.value*(1-neuron3.value))*2(_output[0]-_correctAnswer[cycles])
                _bias1Change[2].append(_b1Change)
                for w in range(5):
                    _w1Change = neuron3.inputs[w]*_memoization
                    _weights1Changes[w+10].append(_w1Change)
            if neuron == 3:
                _w2Change = neuron4.value*_memoization
                _b1Change = -_learningRate*(neuron4.value*(1-neuron4.value))*2(_output[0]-_correctAnswer[cycles])
                _bias1Change[3].append(_b1Change)
                for w in range(5):
                    _w1Change = neuron3.inputs[w]*_memoization
                    _weights1Changes[w+10].append(_w1Change)
            _weights2Change[neuron].append(_w2Change) #placing these values into a list to average them and make a change later
        #calculate the change in bias for the output node
        _b2change = _memoization #unnecessary, but makes the code easier to understand
        _bias2Change.append(_b2change)

#2> Applying the changes to weights and biases
        if cycles%10 == 0: #takes average of weight/bias changes for every 10 cycles
            #first set of weights
            for x in range(20):
                _weights1Changes[x] = np.average(_weights1Changes[x])
            _weights1 = _weights1+_weights1Changes
            _weights1Changes = np.array([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]])
            #first set of biases
            for x in range(4):
                _bias1Change[x] = np.average(_bias1Change[x])
            _bias1 = _bias1+_bias1Change
            _bias1Change = np.array([[],[],[],[]])
            #second set of weights
            for x in range(4):
                _weights2Changes[x] = np.array(_weights2Changes[x])
            _weights2 = _weights2+_weights2Changes
            _weights2Changes = np.array([[],[],[],[]])
            #second bias
            _bias1 = _bias1+np.average(_bias2Change)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#DATA COLLECTION AND FILE MANIPULATION

#reading the csv file for the training data sets and answers, assigning them their respective numpy arrays
def Input_collect():
    global _inputs
    global _correctAnswer
    _tempStore = []
    with open("training_data.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for n in range(5):
                _tempStore.append(row[n])
            _inputs = np.append(_inputs, _tempStore)
            _inputs = _inputs.astype(np.float)
            _correctAnswer = np.append(_correctAnswer, row[5])
            _correctAnswer = _correctAnswer.astype(np.float)
            _tempStore = []

#saving the state of the trained neural net weights and biases
def Save_state():
    np.savetxt('w1.csv', _weights1, delimiter=',')
    np.savetxt('b1.csv', _bias1, delimiter=',')
    np.savetxt('w2.csv', _weights2, delimiter=',')
    np.savetxt('b2.csv', _bias2, delimiter=',')

def Using_saved():
    #Assign the saved weights and biases from their respective files
    _weights1 = np.loadtxt('w1.csv', delimiter=',')
    _bias1 = np.loadtxt('b1.csv', delimiter=',')
    _weights2 = np.loadtxt('w2.csv', delimiter=',')
    _bias2 = np.loadtxt('b2.csv', delimiter=',')
    #also do something to take in inputs
    _inputs = np.loadtxt('i_data.csv', delimiter=',')
    #Process the output guess
    neuron1 = neuron(_weights1[0],_inputs,_bias1[0])
    neuron2 = neuron(_weights1[1],_inputs,_bias1[1])
    neuron3 = neuron(_weights1[2],_inputs,_bias1[2])
    neuron4 = neuron(_weights1[3],_inputs,_bias1[3])
    _hiddenLayer1 = np.array([neuron1.calcSelf(),neuron2.calcSelf(),neuron3.calcSelf(),neuron4.calcSelf()])
    _output = SigmoidFreud(np.sum(np.multiply(_weights2,_hiddenLayer1))+_bias2) #calculation of the output neuron and normalising to produce a number between 0 and 1
    if _output[0] >= 0.5:
        print("\n\nIt will rain in the afternoon :(\nWell I'm at least "+str(_output[0]*100)+"% sure...\n\n")
        quit()
    else:
        print("\n\nIt won't rain :)\nWell I'm at least "+str(100-_output[0]*100)+"% sure...\n\n")
        quit()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#INITIALISING THE PROGRAM

def main():
    choice = input("\n \n \n \ntype '1' to train or type '2' to use already trained:")
    if choice == "1":
        Input_collect()
        train()
        Save_state()
    if choice == "2":
        Using_saved()
    else:
        print("Not a valid input")
        quit()

if __name__ == "__main__":
    main()
