from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, Flatten, Add
from tensorflow.keras.models import Model


class Transformer:
    def __init__(self, shape_of_input, head_size=32, num_heads=2, ff_dim=64, num_transformer_blocks=2, dropout=0.2):
        """
        Initializes the Transformer class with given parameters.

        Parameters:
        - shape_of_input: Tuple, input shape of the data (window_size, features).
        - head_size: Size of each attention head in Multi-Head Attention.
        - num_heads: Number of attention heads.
        - ff_dim: Dimension of the Feed-Forward network inside the transformer block.
        - num_transformer_blocks: Number of stacked transformer blocks.
        - dropout: Dropout rate for regularization.
        """
        self.shape_of_input = shape_of_input
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout = dropout

    def transformer_encoder(self, inputs):
        """
        Defines the transformer encoder block.

        Parameters:
        - inputs: Input layer or data for the transformer encoder.

        Returns:
        - Output of the transformer encoder block after applying Multi-Head Attention and Feed-Forward layers.
        """
        attention_output = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.head_size, dropout=self.dropout)(
            inputs, inputs)
        attention_output = Add()([inputs, attention_output])
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

        ff_output = Dense(self.ff_dim, activation='relu')(attention_output)
        ff_output = Dropout(self.dropout)(ff_output)
        ff_output = Dense(inputs.shape[-1])(ff_output)

        ff_output = Add()([attention_output, ff_output])
        ff_output = LayerNormalization(epsilon=1e-6)(ff_output)

        return ff_output

    def create_transformer_model(self) -> Model:
        """
        Creates the transformer model for NILM.

        Returns:
        - A compiled transformer model ready for training.
        """
        inputs = Input(shape=self.shape_of_input)
        x = inputs

        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x)

        x = Flatten()(x)
        outputs = Dense(1, activation='linear')(x)

        model = Model(inputs, outputs)
        return model
