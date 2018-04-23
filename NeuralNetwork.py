import math as m

eta=0.01

def cross_entropy(expected,value):
    return -m.log(value[expected])

def load_data(f):
    data=[]
    with open(f,"r") as file:
        for line in file:
            data.append([float(field) for field in line.strip().strip()])
            return data

def logistic_sigmoid(x,derivative=False):
    if not derivative:
        d = 1 + m.e**(-(x**2))
        return (2/d)-1
    else:
        return logistic_sigmoid(x)*(1-logistic_sigmoid(x))

class Neuron(object):
    def __init__(self,weight=None,activation=None):
        self.weights=weights
        self.activation=activation

    def activate(self,inputs):

        self.output=[]
        if self.activation is None:
            self.output.extend(inputs)

        else:
            sum=0.0
            for i in range(len(inputs)):
                sum +=inputs[i]*self.weights[i]
            self.output=sum
        return self.output
    def output(self):
        return self.output
    def calculate_delta(self,error):
        self.delta=error*self.activation(self.output,True)

    def update_weights(self,inputs):
        for i in range(len(self.weights)):
            self.weights[i]-=eta*inputs[i]*self.delta

class Layer(object):

    def __init__(self,label=None):

        self.neurons=[]
        self.label=label

        def add_neuron(self,neuron):
            self.neurons.append(neuron)

        def activate(self,inputs):
            if len(self.neurons)==0:
                self.output=[]
                self.output.extend(inputs)

            else:
                self.output=[]
                for neuron in self.neurons:
                    self.output.append(neuron.activate(inputs))

            return self.output

    def output_value(self):
        return self.output

    def softmax(self):
        d = sum([(m.e ** el) for el in self.output])
        return [(m.e ** el) / d for el in self.output]

    def delta(self):
        return [neuron.delta for neuron in self.neurons]

    def calculate_delta(self, next_layer, expected):
        current_layer_errors = []

        if self.label == "output":
            sm = self.softmax()
            for i in range(len(self.neurons)):
                current_layer_errors.append(sm[i] - expected[i])

            else:
                for i in range(len(self.neurons)):
                    tmp = 0.0
                    for j in range(len(next_layer.neurons)):
                        tmp += next_layer.neurons[j].weights[i] * next_layer.neurons[j].delta
                    current_layer_errors.append(tmp)

            for i in range(len(self.neurons)):
                self.neurons[i].calculate_delta(current_layer_errors[i])

            return [neuron.delta for neuron in self.neurons]

    def update_weights(self,inputs):
        for neuron in self.neurons:
                neuron.update_weights(inputs)

    def get_weights(self):
        return [neuron.weights for neuron in self.neurons]



import random

def ran():
    return random.uniform(-1.0,1.0)

def main():
    network=[]

    network=[]
    input_layer=Layer()
    hidden_layer=Layer()
    hidden_layer.add_neuron(Neuron([ran(),ran()],logistic_sigmoid))
    hidden_layer.add_neuron(Neuron([ran(),ran()],logistic_sigmoid))
    hidden_layer.add_neuron(Neuron([ran(),ran()],logistic_sigmoid))
    hidden_layer.add_neuron(Neuron([ran(),ran()],logistic_sigmoid))
    hidden_layer.add_neuron(Neuron([ran(),ran()],logistic_sigmoid))

    output_layer=Layer("output")

    output_layer.add_neuron(Neuron([ran(),ran(),ran(),ran(),ran()],logistic_sigmoid))
    output_layer.add_neuron(Neuron([ran(),ran(),ran(),ran(),ran()],logistic_sigmoid))

    data=load_data("data.txt")

    for i in range(1000):
        for entry in data:
            #forward propogation
            input_layer.activate(entry[:2])
            hidden_layer.activate(input_layer.output_value())
            output_layer.activate(hidden_layer.output_value())

            output_layer.calculate_delta(None,[1,0] if entry[2]==0 else [0,1])
            hidden_layer.calculate_delta(output_layer,None)
            output_layer.update_weights(hidden_layer.output_value())
            hidden_layer.update_weights(entry[:2])

            print(entry[0],entry[1],output_layer.softmax()[0],output_layer.softmax()[1])


main()








