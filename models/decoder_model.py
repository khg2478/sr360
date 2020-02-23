import tensorflow as tf
import config

def conv_layer(net, num_filters, kernel_size, stride_size) :
	net = tf.keras.layers.Conv2D(num_filters, kernel_size,
			padding='same',
			use_bias=False,
			strides=stride_size)(net)
	net = tf.keras.layers.BatchNormalization()(net)
	net = tf.keras.layers.ReLU(6.)(net)
	return net

def get_decoding_model() :
	encoded_input = tf.keras.layers.Input(batch_size=config.BATCH_SIZE,shape=(1024), name="input")
	reshaped_input = tf.keras.layers.Reshape((1, 1, 1024), name='reshaped_input')(encoded_input)
	# net = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(reshaped_input)
	net = conv_layer(reshaped_input, 1024, 3, 1)
	net = tf.keras.layers.Conv2DTranspose(filters=1024, kernel_size=3, strides= 2, padding="valid")(net)
	net = conv_layer(net, 1024, 3, 1)
	net = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=3, strides= 2, padding="valid")(net)
	net = conv_layer(net, 512, 3, 1)
	net = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides= 2, padding="valid")(net)
	net = conv_layer(net, 256, 3, 1)
	net = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides= 2, padding="valid")(net)
	net = conv_layer(net, 256, 3, 1)
	net = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides= 2, padding="valid")(net)
	net = conv_layer(net, 128, 3, 1)
	net = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides= 2, padding="valid")(net)
	net = conv_layer(net, 128, 3, 1)
	net = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides= 2, padding="valid")(net)
	net = conv_layer(net, 64, 3, 1)
	net = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, strides= 2, padding="valid")(net)
	net = conv_layer(net, 3, 3, 1)
	net = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides= 2, padding="valid")(net)
	net = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(net)

	model = tf.keras.Model(inputs=encoded_input, outputs=net)
	return model


