import torch

class RBM():
    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4, use_cuda=True):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda

        ## Inititialize the weights
        self.weights = torch.randn((num_visible, num_hidden)) * 0.1
        ## initialize the forward (encoding) and backward (generating/decoding) biases
        self.visible_bias = torch.ones(num_visible) ## a
        self.hidden_bias = torch.zeros(num_hidden) ## b from the RMB practical guide

        ## training momentum tensors
        self.weight_momentum = torch.zeros((num_visible, num_hidden))
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)

        ## put to GPU if cuda
        if self.use_cuda:
            self.weights = self.weights.cuda()
            self.visible_bias = self.visible_bias.cuda()
            self.hidden_bias = self.hidden_bias.cuda()

            self.weights_momentum = self.weights_momentum.cuda()
            self.visible_bias_momentum = self.visible_bias_momentum.cuda()
            self.hidden_bias_momentum = self.hidden_bias_momentum.cuda()

    ## Forward function to sample a encoded vector by forward pass of visible probabilities (encoding)
    def sample_hidden(self, visible_probabilities):
        ## forward pass for activations in encoding
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        ## Get the sigmoid probabilities for each activation in encoding
        hidden_probabilities = self._sigmoid(hidden_activations)
        return hidden_probabilities

    ## Backward function to sample a visible vector by backward pass of hidden probabilities (generation)
    def sample_visible(self, hidden_probabilities):
        ## backward pass of the hidden probabilities (with the transposed weights)
        visible_activation = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        ## get the corresponding sigmoid probabilities
        visible_probabilities = self._sigmoid(visible_activation)
        return visible_probabilities

    ## Contrastive learning algorithm. refer the RMB practical guide
    def contrastive_divergence(self, input_data):
        ## perform a <v,h>-data and <v,h>-model steps
        ## <v,h>-data step
        positive_hidden_probabilities = self.sample_hidden(input_data)
        ## mask the probabilities with a threshold and get the binary activations
        positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(num_hidden)).float()
        ## get the positive association <v,h>-data
        # matmul gives a dot product of two arrays
        positive_associations = torch.matmul(input_data.t(), positive_hidden_activations)

        ## <v,h>-model step
        ## this is an iterative step that theoritically iterates to inf
        ## here iterate k steps
        ## this is a gibbs markov chain
        # initialize the input to generate from the previous step
        hidden_activations = positive_hidden_activations
        for i in range(self.k):
            ## generate the visible probabilities
            # TODO: Also check what happens when the visible_probabilities are binarized (mentioned in the RBM practical guide pg.5)
            visible_probabilities = self.sample_visible(hidden_activations)
            ## input the visible probabilities and sample an encode of it
            negative_hidden_probabilities = self.sample_hidden(visible_probabilities)
            ## get the binary activation of the hidden sigmoid probabilities
            negative_hidden_activations = (negative_hidden_probabilities >= self._random_probabilities(num_hidden)).float()

        ## get the negative assotions from the last encoded step (this encode is happened through gibbs steps, a encoding and generation loop)
        negative_visible_probabilities = visible_probabilities
        # TODO: change the negative_hidden_activations to negative_hidden_probabilities (mentioned in the RBM practical guide pg.5)
        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_activations)

        ## update the weight and bias momentums
        self.weight_momentum *= self.momentum_coefficient
        self.weight_momentum += (positive_associations - negative_associations)
        ## similar to weights, update the encoding (b) and generating (a) biases
        self.visible_bias_momentum *= self.momentum_coefficient
        ## sum along the batch dim
        self.visible_bias_momentum += torch.sum((input_data - negative_visible_probabilities), dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum((positive_hidden_activations), dim=0)

        ## step the weights by mulitplying the momentums by a normalized learninig rate, normalized by the batch_size
        batch_size = input_data.shape[0]
        self.weights += self.weight_momentum * (self.learning_rate / batch_size)
        ## do the same for the encoding (b) and generating (a) biases
        self.hidden_bias += self.hidden_bias_momentum * (self.learning_rate / batch_size)
        self.visible_bias += self.visible_bias_momentum * (self.learning_rate / batch_size)

        ## perform L2 weight decay; refer to RBM practical guide
        self.weights -= self.weights * self.weight_decay

        ## compte the reconstruction error and return
        error = torch.sum((input_data - negative_visible_probabilities) ** 2)

        return error
        
    ## Compute the sigmoid probs of a input tensor
    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    ## generate a random vector for masking
    def _random_probabilities(self, num):
        ## generate a uniform distribution in the intervel of [0, 1)
        random_probabilities = torch.rand(num)
        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities



################################# CLASS TEST ###############################################
num_visible = 784
num_hidden = 100
k = 1

test_rbm = RBM(num_visible, num_hidden, k, use_cuda=False)
test_input = torch.randn((10, 784)) ## a random normal distribution betweein [0, 1)

## encoding
out = test_rbm.sample_hidden(test_input)
print(out.shape)
## generating
vis_out = test_rbm.sample_visible(out)
print(vis_out.shape)

## step a contrastive_divergence training iter
error = test_rbm.contrastive_divergence(test_input)
print(error)



    

