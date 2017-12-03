import numpy as np

import backend
import nn

def prediction(x, layer_vars):
    assert len(layer_vars) > 0
    relu = np.vectorize(max)
    first_layer = layer_vars[0]
    result = np.dot(x, first_layer[0])+first_layer[1]
    for layer in layer_vars[1:]:
        result = np.dot(relu(result, 0.0), layer[0]) + layer[1]
    return result

def build_n_layer_relu_graph(graph, input_x, layers):
    assert len(layers) > 0
    first_layer = layers[0]
    A = nn.MatrixMultiply(graph, input_x, first_layer[0])
    B = nn.MatrixVectorAdd(graph, A, first_layer[1])
    for layer in layers[1:]:
        C = nn.ReLU(graph, B)
        A = nn.MatrixMultiply(graph, C, layer[0])
        B = nn.MatrixVectorAdd(graph, A, layer[1])
    return B

class Model(object):
    """Base model class for the different applications"""
    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)

class RegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        # (0.093, [20, 21])
        self.learning_rate = 0.093
        self.hidden_size = [20, 21]
        self.var_nodes = dict({"W1": nn.Variable(1, self.hidden_size[0]),
                               "b1": nn.Variable(1, self.hidden_size[0]),
                               "W2": nn.Variable(self.hidden_size[0], self.hidden_size[1]),
                               "b2": nn.Variable(1, self.hidden_size[1]),
                               "W3": nn.Variable(self.hidden_size[1], 1),
                               "b3": nn.Variable(1,1)})
    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        vns = self.var_nodes
        W1 = vns["W1"]
        b1 = vns["b1"] 
        W2 = vns["W2"]
        b2 = vns["b2"] 
        W3 = vns["W3"]
        b3 = vns["b3"] 
        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            layers = [(W1,b1),(W2,b2),(W3,b3)]
            graph = nn.Graph([layers[0][0], layers[0][1],
                              layers[1][0], layers[1][1],
                              layers[2][0], layers[2][1]])
            input_x = nn.Input(graph, x)
            input_y = nn.Input(graph, y)
            input_neg = nn.Input(graph, -np.ones((1,1)))
            
            A = build_n_layer_relu_graph(graph, input_x, layers)
            
            nX = nn.MatrixMultiply(graph, input_x, input_neg)
            B = build_n_layer_relu_graph(graph, nX, layers)
            nB = nn.MatrixMultiply(graph, B, input_neg)

            C = nn.Add(graph, A, nB)
            nn.SquareLoss(graph, C, input_y)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            vnds = [(W1.data,b1.data),(W2.data,b2.data),(W3.data,b3.data)]
            return prediction(x, vnds)-prediction(-x, vnds)

class OddRegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        # (0.093, [20, 21])
        self.learning_rate = 0.093
        self.hidden_size = [20, 21]
        self.var_nodes = dict({"W1": nn.Variable(1, self.hidden_size[0]),
                               "b1": nn.Variable(1, self.hidden_size[0]),
                               "W2": nn.Variable(self.hidden_size[0], self.hidden_size[1]),
                               "b2": nn.Variable(1, self.hidden_size[1]),
                               "W3": nn.Variable(self.hidden_size[1], 1),
                               "b3": nn.Variable(1,1)})
    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        vns = self.var_nodes
        W1 = vns["W1"]
        b1 = vns["b1"] 
        W2 = vns["W2"]
        b2 = vns["b2"] 
        W3 = vns["W3"]
        b3 = vns["b3"] 
        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            layers = [(W1,b1),(W2,b2),(W3,b3)]
            graph = nn.Graph([layers[0][0], layers[0][1],
                              layers[1][0], layers[1][1],
                              layers[2][0], layers[2][1]])
            input_x = nn.Input(graph, x)
            input_y = nn.Input(graph, y)
            input_neg = nn.Input(graph, -np.ones((1,1)))
            
            A = build_n_layer_relu_graph(graph, input_x, layers)
            
            nX = nn.MatrixMultiply(graph, input_x, input_neg)
            B = build_n_layer_relu_graph(graph, nX, layers)
            nB = nn.MatrixMultiply(graph, B, input_neg)

            C = nn.Add(graph, A, nB)
            nn.SquareLoss(graph, C, input_y)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            vnds = [(W1.data,b1.data),(W2.data,b2.data),(W3.data,b3.data)]
            return prediction(x, vnds)-prediction(-x, vnds)

class DigitClassificationModel(Model):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.5
        self.hidden_size = [250]
        self.var_nodes = dict({"W1": nn.Variable(784, self.hidden_size[0]),
                               "b1": nn.Variable(1, self.hidden_size[0]),
                               "W2": nn.Variable(self.hidden_size[0], 10),
                               "b2": nn.Variable(1, 10)})

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        "*** YOUR CODE HERE ***"
        if y is not None:
            "*** YOUR CODE HERE ***"
            graph = nn.Graph([self.var_nodes["W1"],
                              self.var_nodes["b1"],
                              self.var_nodes["W2"],
                              self.var_nodes["b2"]])
            input_x = nn.Input(graph, x)
            input_y = nn.Input(graph, y)

            A = nn.MatrixMultiply(graph, input_x, self.var_nodes['W1'])
            B = nn.MatrixVectorAdd(graph, A, self.var_nodes["b1"])
            C = nn.ReLU(graph, B)
            D = nn.MatrixMultiply(graph, C, self.var_nodes["W2"])
            E = nn.MatrixVectorAdd(graph, D, self.var_nodes["b2"])
            nn.SoftmaxLoss(graph, E, input_y)
            return graph
        else:
            "*** YOUR CODE HERE ***"
            vns = self.var_nodes
            W1 = vns["W1"].data
            b1 = vns["b1"].data
            W2 = vns["W2"].data
            b2 = vns["b2"].data
            return prediction(x, [(W1,b1),(W2,b2)])

class DeepQModel(Model):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.005
        self.hidden_size = [20, 21]
        self.var_nodes = dict({"W1": nn.Variable(self.state_size, self.hidden_size[0]),
                               "b1": nn.Variable(1, self.hidden_size[0]),
                               "W2": nn.Variable(self.hidden_size[0], self.hidden_size[1]),
                               "b2": nn.Variable(1, self.hidden_size[1]),
                               "W3": nn.Variable(self.hidden_size[1], self.num_actions),
                               "b3": nn.Variable(1,self.num_actions)})

    def run(self, states, Q_target=None):
        """
        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"
        vns = self.var_nodes
        W1 = vns["W1"]
        b1 = vns["b1"] 
        W2 = vns["W2"]
        b2 = vns["b2"] 
        W3 = vns["W3"]
        b3 = vns["b3"] 
        if Q_target is not None:
            "*** YOUR CODE HERE ***"
            layers = [(W1,b1),(W2,b2),(W3,b3)]
            graph = nn.Graph([layers[0][0], layers[0][1],
                              layers[1][0], layers[1][1],
                              layers[2][0], layers[2][1]])
            in_state = nn.Input(graph, states)
            in_target = nn.Input(graph, Q_target)
            #input_neg = nn.Input(graph, -np.ones((1,1)))
            
            A = build_n_layer_relu_graph(graph, in_state, layers)
            nn.SquareLoss(graph, A, in_target)
            return graph
        else:
            "*** YOUR CODE HERE ***"
            return prediction(states, [(W1.data,b1.data),(W2.data,b2.data),(W3.data,b3.data)])

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.007
        self.hidden_size = [150]
        self.var_nodes = dict({"W1": nn.Variable(self.num_chars, self.hidden_size[0]),
                               "b1": nn.Variable(1, self.hidden_size[0]),
                               "W2": nn.Variable(self.hidden_size[0], self.num_chars),
                               "b2": nn.Variable(1, self.num_chars),
                               "W0": nn.Variable(self.num_chars, len(self.languages)),
                               "b0": nn.Variable(1, len(self.languages))})


    def run(self, xs, y=None):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """
        batch_size = xs[0].shape[0]
        "*** YOUR CODE HERE ***"
        vns = self.var_nodes
        W0 = vns["W0"]
        b0 = vns["b0"]
        layers = [(vns["W1"],vns["b1"]),(vns["W2"],vns["b2"])]
        graph = nn.Graph([layers[0][0], layers[0][1],
                          layers[1][0], layers[1][1],
                          W0, b0])
        xns = [nn.Input(graph, x) for x in xs]
        A = nn.Input(graph, np.zeros_like(xs[0]))
        for xn in xns:
            A = build_n_layer_relu_graph(graph, nn.Add(graph, A, xn), layers)
        A = build_n_layer_relu_graph(graph, A, [(W0, b0)])

        if y is not None:
            "*** YOUR CODE HERE ***"
            in_y = nn.Input(graph, y)
            nn.SoftmaxLoss(graph, A, in_y)
            return graph
        else:
            "*** YOUR CODE HERE ***"
            return graph.get_output(A)

