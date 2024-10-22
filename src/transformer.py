from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, Add, \
    GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import TimeDistributed


class Transformer:
    def __init__(self, shape_of_input, head_size=32, num_heads=2, ff_dim=64, num_transformer_blocks=2, dropout=0.2):
        self.shape_of_input = shape_of_input
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout = dropout

    def transformer_encoder(self, inputs):
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

    def transformer_decoder(self, inputs, encoder_output):
        attention_output = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.head_size, dropout=self.dropout)(
            inputs, encoder_output)
        attention_output = Add()([inputs, attention_output])
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

        ff_output = Dense(self.ff_dim, activation='relu')(attention_output)
        ff_output = Dropout(self.dropout)(ff_output)
        ff_output = Dense(inputs.shape[-1])(ff_output)

        ff_output = Add()([attention_output, ff_output])
        ff_output = LayerNormalization(epsilon=1e-6)(ff_output)

        return ff_output

    def create_transformer_model(self, metrics=None, num_appliances=None):
        inputs = Input(shape=self.shape_of_input)
        x = inputs

        # Encoder
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x)

        # Decoder
        decoder_inputs = Input(shape=(None, self.shape_of_input[-1]))  # Adjust shape as necessary
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_decoder(decoder_inputs, x)

        # Final layers
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        outputs = TimeDistributed(Dense(num_appliances, activation='linear')(x))  # Adjust for number of appliances

        model = Model([inputs, decoder_inputs], outputs)
        return model

