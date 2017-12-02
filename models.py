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
        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            graph = nn.Graph([self.var_nodes["W1"],
                              self.var_nodes["b1"],
                              self.var_nodes["W2"],
                              self.var_nodes["b2"],
                              self.var_nodes["W3"],
                              self.var_nodes["b3"]])
            input_x = nn.Input(graph, x)
            input_y = nn.Input(graph, y)
            input_neg = nn.Input(graph, -np.ones((1,1)))
            
            A = nn.MatrixMultiply(graph, input_x, self.var_nodes['W1'])
            B = nn.MatrixVectorAdd(graph, A, self.var_nodes["b1"])
            C = nn.ReLU(graph, B)
            D = nn.MatrixMultiply(graph, C, self.var_nodes["W2"])
            E = nn.MatrixVectorAdd(graph, D, self.var_nodes["b2"])
            F = nn.ReLU(graph, E)
            G = nn.MatrixMultiply(graph, F, self.var_nodes["W3"])
            H = nn.MatrixVectorAdd(graph, G, self.var_nodes["b3"])
            
            nX = nn.MatrixMultiply(graph, input_x, input_neg)
            I = nn.MatrixMultiply(graph, nX, self.var_nodes['W1'])
            J = nn.MatrixVectorAdd(graph, I, self.var_nodes["b1"])
            K = nn.ReLU(graph, J)
            L = nn.MatrixMultiply(graph, K, self.var_nodes["W2"])
            M = nn.MatrixVectorAdd(graph, L, self.var_nodes["b2"])
            N = nn.ReLU(graph, M)
            O = nn.MatrixMultiply(graph, N, self.var_nodes["W3"])
            P = nn.MatrixVectorAdd(graph, O, self.var_nodes["b3"])
            nP = nn.MatrixMultiply(graph, P, input_neg)

            Q = nn.Add(graph, H, nP)
            loss = nn.SquareLoss(graph, Q, input_y)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            vns = self.var_nodes
            W1 = vns["W1"].data
            b1 = vns["b1"].data
            W2 = vns["W2"].data
            b2 = vns["b2"].data
            W3 = vns["W3"].data
            b3 = vns["b3"].data
            relu = np.vectorize(max)
            A1 = relu(np.dot(x, W1) + b1, 0.0)
            A2 = relu(np.dot(A1,W2) + b2, 0.0)
            A3 = np.dot(A2,W3) + b3
            C1 = relu(np.dot(-x,W1) + b1, 0.0)
            C2 = relu(np.dot(C1,W2) + b2, 0.0)
            C3 = np.dot(C2,W3) + b3
            return A3 - C3

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
        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            graph = nn.Graph([self.var_nodes["W1"],
                              self.var_nodes["b1"],
                              self.var_nodes["W2"],
                              self.var_nodes["b2"],
                              self.var_nodes["W3"],
                              self.var_nodes["b3"]])
            input_x = nn.Input(graph, x)
            input_y = nn.Input(graph, y)
            input_neg = nn.Input(graph, -np.ones((1,1)))
            
            A = nn.MatrixMultiply(graph, input_x, self.var_nodes['W1'])
            B = nn.MatrixVectorAdd(graph, A, self.var_nodes["b1"])
            C = nn.ReLU(graph, B)
            D = nn.MatrixMultiply(graph, C, self.var_nodes["W2"])
            E = nn.MatrixVectorAdd(graph, D, self.var_nodes["b2"])
            F = nn.ReLU(graph, E)
            G = nn.MatrixMultiply(graph, F, self.var_nodes["W3"])
            H = nn.MatrixVectorAdd(graph, G, self.var_nodes["b3"])
            
            nX = nn.MatrixMultiply(graph, input_x, input_neg)
            I = nn.MatrixMultiply(graph, nX, self.var_nodes['W1'])
            J = nn.MatrixVectorAdd(graph, I, self.var_nodes["b1"])
            K = nn.ReLU(graph, J)
            L = nn.MatrixMultiply(graph, K, self.var_nodes["W2"])
            M = nn.MatrixVectorAdd(graph, L, self.var_nodes["b2"])
            N = nn.ReLU(graph, M)
            O = nn.MatrixMultiply(graph, N, self.var_nodes["W3"])
            P = nn.MatrixVectorAdd(graph, O, self.var_nodes["b3"])
            nP = nn.MatrixMultiply(graph, P, input_neg)

            Q = nn.Add(graph, H, nP)
            loss = nn.SquareLoss(graph, Q, input_y)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            vns = self.var_nodes
            W1 = vns["W1"].data
            b1 = vns["b1"].data
            W2 = vns["W2"].data
            b2 = vns["b2"].data
            W3 = vns["W3"].data
            b3 = vns["b3"].data
            relu = np.vectorize(max)
            A1 = relu(np.dot(x, W1) + b1, 0.0)
            A2 = relu(np.dot(A1,W2) + b2, 0.0)
            A3 = np.dot(A2,W3) + b3
            C1 = relu(np.dot(-x,W1) + b1, 0.0)
            C2 = relu(np.dot(C1,W2) + b2, 0.0)
            C3 = np.dot(C2,W3) + b3
            return A3 - C3

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
        # (0.230, 160) {96.90, 95.59}
        # (0.230, 150) {97.10, 92.88, 92.83}
        # (0.270, 125) {96.46, 96.97, 96.79}
        # (0.250, 125) {96.71, 95.90}
        # (0.230, 125) {93.80, 95.77}
        # (0.240, 125) {96.45, 96.08}
        # (0.020, 250) {90.45, }
        # (0.040, 250) {94.35, }
        # (0.080, 250) {95.20, }
        # (0.160, 250) {96.13, 96.28}
        # (0.180, 250) {96.74, 94.70}
        # (0.220, 250) {95.56, 96.72}
        # (0.240, 250) {97.07, 96.97, 96.77, 97.17, 97.01, 96.80}
        # (0.250, 250) {96.39, 95.51}
        # (0.230, 250) {97.22, 96.47, 92.13}
        # (0.240, 175) {95.43}
        # (0.239, 250) {96.83, 97.47, 97.20}
        self.learning_rate = 0.239
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
            F = nn.SoftmaxLoss(graph, E, input_y)
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

        if Q_target is not None:
            "*** YOUR CODE HERE ***"
        else:
            "*** YOUR CODE HERE ***"

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

        if y is not None:
            "*** YOUR CODE HERE ***"
        else:
            "*** YOUR CODE HERE ***"
