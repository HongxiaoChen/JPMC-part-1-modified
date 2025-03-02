import tensorflow as tf


class MLP(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, latent_dim=None, nonlinearity='sine'):
        """
        Args:
            input_dim
            hidden_dim
            latent_dim
            nonlinearity: activation
        """
        super(MLP, self).__init__()

        self.latent_dim = latent_dim if latent_dim is not None else input_dim

        if nonlinearity == 'sine':
            self.nonlinearity = tf.math.sin
        elif nonlinearity == 'tanh':
            self.nonlinearity = tf.nn.tanh
        elif nonlinearity == 'relu':
            self.nonlinearity = tf.nn.relu
        else:
            raise ValueError(f"Nonlinearity {nonlinearity} not recognized")

        
        self.layer1 = tf.keras.layers.Dense(hidden_dim)
        self.layer2 = tf.keras.layers.Dense(hidden_dim)
        self.layer3 = tf.keras.layers.Dense(self.latent_dim)

    def call(self, x):
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            Latent variable: [batch_size, latent_dim]
        """
        h = self.nonlinearity(self.layer1(x))
        h = self.nonlinearity(self.layer2(h))
        return self.layer3(h)
