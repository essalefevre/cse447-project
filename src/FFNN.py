class FFNN(nn.Module): # We inherit from nn.Module, which is the base class for all PyTorch Neural Network modules

    def __init__(
        self, input_dim: int, hidden_dim: int
    ):
        """
        Define the architecture of a Feedforward Neural Network with architecture described above.

        Inputs:
        - input_dim: The dimension of the input (d according to the figure above)
        - hidden_dim: The dimension of the hidden layer (h according to the figure above)
        - num_classes: The number of classes in the classification task.

        """

        super(FFNN, self).__init__() # Call the base class constructor

        # Define your network architecture below

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 3)

        self.initialize_weights() # Initialize the weights of the linear layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Computes the forward pass through the network.

        Inputs:
        - x : Input tensor of shape (n, d) where n is the number of samples and d is the dimension of the input

        Hint: You can call a layer directly with the input to get the output tensor, e.g. self.fc1(x) will return the output tensor after applying the first linear layer.
        """

        return self.fc2(self.act1(self.fc1(x)))

    def initialize_weights(self):
        """
        Initialize the weights of the linear layers.

        We initialize the weights using Xavier Normal initialization and the biases to zero.

        You can read more about Xavier Initialization here: https://cs230.stanford.edu/section/4/#xavier-initialization
        """
        for layer in self.children():
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)