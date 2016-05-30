# -*- coding: utf-8 -*-
import gensim, codecs, numpy, collections
from prettytable import PrettyTable
numpy.random.seed(0)


def sigmoid(number):
    return 1.0 / (1.0 + numpy.exp(-number))
## 1/(1+e^-x)

def sigmoid_prim(number):
    return sigmoid(number) * (1.0 - sigmoid(number))

def tanh(number):
    return (numpy.exp(number) - numpy.exp(-number)) / (numpy.exp(number) + numpy.exp(-number))
## (e^x - e^-x)/(e^x + e^-x)

def tanh_prim(number):
    return 1.0 - numpy.power(tanh(number), 2)
## (4e^(2*x))/(e^(2*x)+1)^2 = 1-((e^x-e^-x)/(e^x+e^-x))^2
    
class Network(object):
    def __init__(self, sizev, contextv):
        self.size = sizev
        self.size[0] += (contextv * (size[0] + (size[-1] + 2))) + (contextv * self.size[0])
        self.biases = []
        self.weights = []
        self.context  = contextv
        self.set_bias()
        self.set_weights()
        self.vec_sigmoid = numpy.vectorize(sigmoid)
        self.vec_sigmoid_prim = numpy.vectorize(sigmoid_prim)


    def set_bias(self):
        for num_nodes in self.size[1:]:
            self.biases.append(numpy.zeros((num_nodes, 1)))

    def set_weights(self):
        counter = 1
        layers = len(self.size)
        connections = []
        while layers > counter:
            tuplev = (self.size[counter], self.size[counter-1])
            connections.append(tuplev)
            counter +=1
        for con in connections:
            self.weights.append(0.2 * numpy.random.randn(con[0], con[1]))
    
    def feedforward(self, inputv):
        vec_sig = numpy.vectorize(sigmoid)
        for bias, weight in zip(self.biases, self.weights):
            inputv = vec_sig(numpy.dot(weight, inputv) + bias)
        return inputv

    def backpropagation(self, inputv, targetv):
        weighted_inputs = []
        activation_values = []
        activation_values.append(inputv)
        for bias, weight in zip(self.biases, self.weights):
            weighted_input = numpy.dot(weight, inputv) + bias
            activation_value = self.vec_sigmoid(weighted_input)
            weighted_inputs.append(weighted_input)
            activation_values.append(activation_value)
            inputv = activation_value
        output_error = (activation_values[-1] - targetv) * self.vec_sigmoid_prim(weighted_inputs[-1])
        error_weight = []
        error_bias = []
        error_weight.append(numpy.dot(output_error, activation_values[-2].T))
        error_bias.append(output_error)
        error = output_error
        for index in xrange(2, len(self.size)):
            error = numpy.dot(self.weights[-index+1].T, error) * self.vec_sigmoid_prim(weighted_inputs[-index])
            error_bias.append(error)
            error_weight.append(numpy.dot(error, activation_values[-index-1].T))
        return (error_bias[::-1], error_weight[::-1])
    
    def update(self, training_ex, learning_rate):
        error_bias = []
        error_weight = []
        for bias in self.biases:
            error_bias.append(numpy.zeros(bias.shape, dtype=numpy.float32))
        for weight in self.weights:
            error_weight.append(numpy.zeros(weight.shape, dtype=numpy.float32))
        for sentence in training_ex:
            for inputv, outputv in sentence:
                delta_error_bias, delta_error_weight = self.backpropagation(inputv, outputv)
                error_bias = [eb + deb for eb, deb in zip(error_bias, delta_error_bias)]
                error_weight = [ew + dew for ew, dew in zip(error_weight, delta_error_weight)]
            self.biases = [bias - (learning_rate / len(training_ex)) * eb for bias, eb in zip(self.biases, error_bias)]
            self.weights = [weight - (learning_rate / len(training_ex)) * ew for weight, ew in zip(self.weights, error_weight)]

    def evaluate(self, data):
        datalength = 0
        correct = 0
        total = 0
        pos_list = collections.OrderedDict()
        pos = ['JJ', 'NN', 'PP', 'PM', 'DT','AB','VB','KN','PS','PC','PN','RG','SN','HP','HD','PL','IE','HA','RO','HS','IN','UO','MAD','PAD','MID']

        for sentence in data:
            datalength += len(sentence)

        print "Evaluating with " + str(datalength) + " testsets."
        for part_of_speech in pos:
            pos_list[part_of_speech] = collections.OrderedDict()
            for part in pos:
                pos_list[part_of_speech][part] = 0
        
        for sentence in data:
            context = set_context(sentence, self.context)
            word_counter = 0
            while word_counter < len(sentence):
                if pure_letters:
                    vec = letter_input(sentence[word_counter], context, self.size[0], self.size[-1])[0]
                else:
                    vec = vectorize_input(sentence[word_counter], context, self.size[0], self.size[-1])[0]
                target = sentence[word_counter][1]
                guess = numpy.argmax(self.feedforward(vec))
                update_context(context, self.context, sentence, word_counter, guess=pos[guess])

                pos_list[pos[guess]][target] += 1
                word_counter += 1
                if guess == pos.index(target):
                    correct += 1

        tablehead = [''] + pos
        table = PrettyTable(tablehead)
        for part_of_speech in pos_list:
            tablerow = [part_of_speech]
            for part in pos_list[part_of_speech]:
                    tablerow += [pos_list[part_of_speech][part]]
            table.add_row(tablerow)
            tablerow = []
        print table
        result = (float(correct)/float(total)) * 100
        print "correct:", correct, "\n", "total:", total, "\n", result


    def mini_eval(self, data):
        correct = 0.0
        total = 0.0
        pos = ['JJ', 'NN', 'PP', 'PM', 'DT','AB','VB','KN','PS','PC','PN','RG','SN','HP','HD','PL','IE','HA','RO','HS','IN','UO','MAD','PAD','MID']
        for sentence in data:
            word_counter = 0
            context = set_context(sentence, self.context)
            while word_counter < len(sentence):
                if pure_letters:
                    inputvector = letter_input(sentence[word_counter], context, self.size[0], self.size[-1])[0]
                else:
                    inputvector = vectorize_input(sentence[word_counter], context, self.size[0], self.size[-1])[0]
                target = sentence[word_counter][1]
                guess = numpy.argmax(self.feedforward(inputvector))
                update_context(context, self.context, sentence, word_counter, guess=pos[guess])
                if guess == pos.index(target):
                    correct += 1
                total += 1
                word_counter += 1
        return correct/total
        
def set_context(sentence, n_context):
    context = [[], []]
    for iter in xrange(n_context):
        context[0].append(("", "BOS"))
        try:
            context[1].append((sentence[iter+1][0], "word"))
        except:
            context[1].append(("", "word"))
    return context


def update_context(present_context, n_context, sentence, word_counter, guess=None):
    context = present_context
    if len(context[0]) == 0:
        return context
    context[0].pop(0)
    if guess == None:
        context[0].insert(n_context, sentence[word_counter])
    else:
        context[0].insert(n_context, (sentence[word_counter][0], guess))
    context[1].pop(0)
    try:
        context[1].insert(n_context, (sentence[word_counter + n_context + 1][0], "word"))
    except:
        context[1].insert(n_context, ("", "word"))
    return context


def train(network, data, n_epochs, batch_size, learning_rate):
    data_length = 0
    counter = 0
    epoch_counter = 0
    evaluation_set = test_data

    for sentence in data:
        data_length += len(sentence)
    for epoch in xrange(n_epochs):
        epoch_counter += 1
        procent = 0
        result = network.mini_eval(evaluation_set) * 100
        print "Epoch " + str(epoch_counter) + ".", "0% data processed. Accuracy: " + str(result) + "%"
        numpy.random.shuffle(data)
        for k in xrange(0, data_length, batch_size):
            mini_batch = data[k:k+batch_size]
            new_batch = []
            sentence_data = []
            for sentence in mini_batch:
                context = set_context(sentence, network.context)
                word_counter = 0
                while word_counter < len(sentence):
                    if pure_letters:
                        sentence_data.append(letter_input(sentence[word_counter], context, network.size[0], network.size[-1]))
                    else:
                        sentence_data.append(vectorize_input(sentence[word_counter], context, network.size[0], network.size[-1]))
                    context = update_context(context, network.context, sentence, word_counter)
                    word_counter += 1
                    counter += 1
                    if (counter % int(data_length/100)) == 0:
                        procent += 1
                        result = network.mini_eval(evaluation_set) * 100
                        print "Epoch " + str(epoch_counter) + ".", str(procent) + "% data processed. Accuracy: " + str(result) + "%"
                new_batch.append(sentence_data)
                sentence_data = []
            network.update(new_batch, learning_rate)


def load_data(data, model):
    data_set = []
    data = codecs.open(data, 'r', 'utf-8')
    sentence = []
    for line in data:
        dataline = line.split('\t')
        if dataline[0] == "\n":
            data_set.append(sentence)
            sentence = []
        else:
            if pure_letters:
                if len(dataline[0]) > 30:
                    pass
                else:
                    target = dataline[1][:-1]
                    word = dataline[0]
                    sentence.append((word, target))
            else:
                target = dataline[1][:-1]
                word = dataline[0]
                sentence.append((word, target))
    return data_set

def vectorize_input(inputv, context, inputlength, outputlength):
    context_vector = []
    for con in context[0]:
        if con[1] == "BOS":
            context_vector += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            for iter in xrange(300):
                context_vector += [0]
        elif con[1] == "EOS":
            context_vector += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            for iter in xrange(300):
                context_vector += [0]
        else:
            context_vector = numpy.append(context_vector, (pos_dict[con[1]] + [0, 0]))
            context_vector = numpy.append(context_vector, model[con[0]])

    for con in context[1]:
        if con[0] == "":
            for iter in xrange(300):
                context_vector = numpy.append(context_vector, 0)
        else:
            context_vector = numpy.append(context_vector, model[con[0]])

            
    input_vector = numpy.append(model[inputv[0]], context_vector)
    target = numpy.reshape(numpy.array(pos_dict[inputv[1]]), (outputlength,1))
    inputvector = numpy.reshape(numpy.array(input_vector, dtype=float), (len(input_vector), 1))
    return (inputvector, target)
               

pos_dict = {'JJ': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'NN': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'PP': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'PM': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'DT': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'AB': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'VB': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'KN': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'PS': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'PC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'PN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'RG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'SN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'HP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'HD': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'PL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'IE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'HA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'RO': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'HS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'IN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'UO': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'MAD': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'PAD': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'MID': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

## Settings
#Files
model_file = '300features_1minwords_10context'
training_file = 'suc-train.txt'
test_file = 'suc-test.txt'
#Network size
input_data_size = 300 #930/300
hidden_layer_length = 2
hidden_layer_width = 40
output_layer_size = 25
context_size = 1
#Training options
pure_letters = False
epochs = 10
sample_size = 50
learning_rate = 0.01 # 0.01

print "Loading model..."
model = gensim.models.Word2Vec.load(model_file)
print "Done!\n"
size = [input_data_size] + [hidden_layer_width for iter in xrange(hidden_layer_length)] + [output_layer_size]
print "Creating network with the size " + str(size) + " and context length " + str(context_size) + "..."
net = Network(size, context_size)
print "Done!\n"
print "Loading data..."
data = load_data(training_file, model)
test_data = load_data(test_file, model)
print "Done!\n"
print "Training the network..."
train(net, data, epochs, sample_size, learning_rate)
print "Done!\n"
print "Evaluating the network..."
net.evaluate(test_data)

