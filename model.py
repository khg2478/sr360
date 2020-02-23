import tensorflow as tf
import config
from models.keras_applications.mobilenet_output_stride_posenet import MobileNet as MobileNet_PoseNet
from models.decoder_model import *


def get_backbone_model(type_backbone, image_input) :
	if (type_backbone == "ResNet101V2") :
		backbone_model = tf.keras.applications.ResNet101V2(include_top=False, weights='imagenet')(image_input)
	elif (type_backbone == "MobileNetV1_Keras") :
		backbone_model = tf.keras.applications.MobileNet(include_top=False, weights='imagenet')(image_input)
	elif (type_backbone == "MobileNetV2_Keras") :
		backbone_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet')(image_input)
	elif (type_backbone == "MobileNetV1_PoseNet") :
		backbone_model = MobileNet_PoseNet(input_shape=(config.IMAGE_HEIGHT,config.IMAGE_WIDTH,config.IMAGE_CHANNEL), include_top=False, weights=None, output_stride_value=config.OUTPUT_STRIDE, backend=tf.keras.backend, layers=tf.keras.layers, 
			models=tf.keras.models, utils=tf.keras.utils)
		backbone_model.summary()
		backbone_model = backbone_model(image_input)
	return backbone_model

def get_encoding_model(type_backbone) :
	image_input = tf.keras.layers.Input(batch_size=config.BATCH_SIZE,shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3), name="input")
	net = get_backbone_model(type_backbone, image_input)
	encoded_output = tf.keras.layers.GlobalAveragePooling2D()(net)
	#generate model
	model = tf.keras.Model(inputs=image_input, outputs=encoded_output)
	return model

# type_backbone = "MobileNetV1_PoseNet"
# encoding_model = get_encoding_model(type_backbone)

