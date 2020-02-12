#Basic Neural Network Implementation From Scratch
import numpy as np
import scipy.special
import  matplotlib.pyplot as plt

#Neural Network
class NeuralNetwork:
    #Declare Neurons, Synapses, Learning Rate, Activation Functions
    def __init__(self, input, hidden, output, learning_rate):
        #Neurons
        self.input = input
        self.hidden = hidden
        self.output = output

        #Synapses
        self.WIH = np.random.normal(0.0, pow(self.hidden, -(1/2)), (self.hidden, self.input))
        self.WHO = np.random.normal(0.0, pow(self.output, -(1/2)), (self.output, self.hidden))

        #Learning Rate
        self.learning_rate = learning_rate

        #Activation Function
        self.sigmoid = lambda x: scipy.special.expit(x)
        pass
    #Provide Features and Labeled Data and Train the Neural Network
    def train(self, input_list, target_list):
        #Input Layer and Targets (no need to pass through an activation function)
        input_list = np.array(input_list).reshape(-1, 1)
        target_list = np.array(target_list).reshape(-1, 1)

        #Hidden Layer
        hidden_input = np.dot(self.WIH, input_list)
        hidden_out = self.sigmoid(hidden_input)

        #Output Layer
        output_in = np.dot(self.WHO, hidden_out)
        output_out = self.sigmoid(output_in)

        #Error (Output)
        output_error = target_list - output_out
        #Update Weights between Hidden and Output 
        self.WHO += self.learning_rate * np.dot((output_error * output_out * (1 - output_out)), hidden_out.T)
         #Hidden Error
        hidden_error = np.dot(self.WHO.T, output_error)

        #Update Weights between Input and Hidden
        self.WIH += self.learning_rate * np.dot((hidden_error * hidden_out * (1 - hidden_out)), input_list.T)
        pass
    #Provide Input Data and Display Output
    def query(self, input_list):
        #Input Layer (no need to pass through an activation function)
        input_list = np.array(input_list).reshape(-1, 1)

        #Hidden Layer
        hidden_input = np.dot(self.WIH, input_list)
        hidden_out = self.sigmoid(hidden_input)

        #Output Layer
        output_in = np.dot(self.WHO, hidden_out)
        output_out = self.sigmoid(output_in)

        #Return Output
        return (output_out)
    pass


#Create Neural Network Object from Class
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#Open Database and Store Content on data_list
with open('mnist_train_100.csv', 'r') as training_data_file:
    traing_data_list = training_data_file.readlines()

#Train Network
for record in traing_data_list:
    #Split Data
    all_values = record.split(',')
    #Scale Input
    scaled_input = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
    #Expected Output
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    neural_network.train(scaled_input, targets)
    pass

#Test Network
with open('mnist_test_10.csv', 'r') as test_data_file:
    test_data_list = test_data_file.readlines()

#Test Label
all_values = test_data_list[0].split(',')
print('Expected large float on position: ' + all_values[0])

#Create a Graph
image_array = np.asfarray(all_values[1:]).reshape(28,28)
plt.imshow(image_array, cmap = 'Greys', interpolation = 'None')

#Query Neural Network
output = neural_network.query((np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01)

print(output)
plt.show()