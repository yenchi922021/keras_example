from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.layers import Lambda
from keras import backend as K

def seq2seq(num_encoder_tokens, latent_dim, num_decoder_tokens, max_decoder_seq_length):
	# The first part is unchanged
	encoder_inputs = Input(shape=(None, num_encoder_tokens), name="encoder_inputs")
	encoder = LSTM(latent_dim, return_state=True, name="encoder")
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	states = [state_h, state_c]

	# Set up the decoder, which will only process one timestep at a time.
	decoder_inputs = Input(shape=(1, num_decoder_tokens), name="decoder_inputs")
	decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder")
	decoder_dense = Dense(num_decoder_tokens, activation='softmax', name="predict")

	all_outputs = []
	inputs = decoder_inputs
	for _ in range(max_decoder_seq_length):
		# Run the decoder on one timestep
		outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)
		outputs = decoder_dense(outputs)
		# Store the current prediction (we will concatenate all predictions later)
		all_outputs.append(outputs)
		# Reinject the outputs as inputs for the next loop iteration
		# as well as update the states
		inputs = outputs
		states = [state_h, state_c]

	# Concatenate all predictions
	decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

	# Define and compile model as previously
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

	return model

if __name__ == "__main__":
	num_encoder_tokens = 300
	latent_dim = 256
	num_decoder_tokens = 8000
	max_decoder_seq_length = 40
	model = seq2seq(num_encoder_tokens, latent_dim, num_decoder_tokens, max_decoder_seq_length)
	model.summary()


