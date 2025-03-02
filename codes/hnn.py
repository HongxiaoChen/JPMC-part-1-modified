import tensorflow as tf


class HNN(tf.keras.Model):
    """
    Hamiltonian Neural Network (HNN) model for learning dynamics based on Hamiltonian mechanics.

    This model computes the Hamiltonian, kinetic energy, and time derivatives of a system
    using a differentiable model provided by the user.
    """

    def __init__(self, input_dim, differentiable_model, mass_matrix=None):
        """
        Initialize the Hamiltonian Neural Network.

        Args:
            input_dim (int): The input dimensionality of the system (q and p combined).
            differentiable_model (callable): A differentiable model (e.g., a neural network)
                that outputs the latent variables for the Hamiltonian.
            mass_matrix (tf.Tensor, optional): Mass matrix for the system, with shape [dim].
                If not provided, defaults to a vector of ones.
        """
        super(HNN, self).__init__()
        self.dim = input_dim // 2
        self.differentiable_model = differentiable_model

        if mass_matrix is None:
            self.M = tf.ones(self.dim)
        else:
            self.M = mass_matrix

    def kinetic_energy(self, p):
        """
        Compute the kinetic energy K(p) = Σ(p² / 2m).

        Args:
            p (tf.Tensor): Momentum tensor with shape [batch_size, dim].

        Returns:
            tf.Tensor: Kinetic energy for each sample in the batch, with shape [batch_size].
        """
        return tf.reduce_sum(0.5 * tf.square(p) / self.M, axis=1)

    def call(self, x):
        """
        Compute the Hamiltonian by summing over all latent variables.
        """

        latent_vars = self.differentiable_model(x)
        return tf.reduce_sum(latent_vars, axis=1, keepdims=True)

    @tf.function
    def compute_hamiltonian(self, x):
        return self.call(x)
