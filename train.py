import config
from model import *

from FrameDataGenerator import FrameDataGenerator

def get_psnr_loss(decoded_imgs, true_data) :
	psnr = tf.image.psnr(decoded_imgs, true_data, max_val=255)
	print("psnr",psnr)
	return psnr

@tf.function
def train_step(encoding_model, decoding_model, optimizer_enc, optimizer_dec, batch_data) :
	with tf.GradientTape() as tape:
		batch_imgs = batch_data[0]
		true_data = batch_data[1:]
		encoded_data = encoding_model(batch_imgs, training=True)
		decoded_imgs = decoding_model(encoded_data, training=True)
		print("encoded_data.shape",encoded_data.shape)

		psnr_loss = get_psnr_loss(decoded_imgs, true_data)
	grads = tape.gradient(psnr_loss, decoding_model.trainable_variables)
	optimizer_dec.apply_gradients(zip(grads, decoding_model.trainable_variables))
	return psnr_loss

def train() :
	print("config",config)
	ckpt_enc_path = config.SAVE_MODEL_PATH + config.BACKBONE_MODEL + "/enc"
	ckpt_dec_path = config.SAVE_MODEL_PATH + config.BACKBONE_MODEL + "/dec"
	encoding_model = get_encoding_model(config.BACKBONE_MODEL)
	encoding_model.summary()
	decoding_model = get_decoding_model()
	decoding_model.summary()


	optimizer_enc = tf.keras.optimizers.Adam(config.LEARNING_RATE)
	optimizer_dec = tf.keras.optimizers.Adam(config.LEARNING_RATE)
	frame_data_generator = FrameDataGenerator()
	ckpt_enc = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer_enc, net=encoding_model)
	ckpt_dec = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer_dec, net=decoding_model)
	manager_enc = tf.train.CheckpointManager(ckpt_enc, ckpt_enc_path, max_to_keep=3)
	manager_dec = tf.train.CheckpointManager(ckpt_dec, ckpt_dec_path, max_to_keep=3)
	ckpt_enc.restore(manager_enc.latest_checkpoint)
	ckpt_dec.restore(manager_dec.latest_checkpoint)
	if manager_enc.latest_checkpoint:
		print("Restored from {}".format(manager_enc.latest_checkpoint))
	else:
		print("Initializing from scratch.")
	if manager_dec.latest_checkpoint:
		print("Restored from {}".format(manager_dec.latest_checkpoint))
	else:
		print("Initializing from scratch.")

	for epoch in range(config.NUM_EPOCHS) :
		for step, batch_data in enumerate(frame_data_generator) :
			total_loss = train_step(encoding_model, decoding_model, optimizer_enc, optimizer_dec, batch_data)
			print("total_loss",total_loss)
			ckpt_enc.step.assign_add(1)
			ckpt_dec.step.assign_add(1)
			if (int(ckpt_enc.step) % 500 == 0):
				save_enc_path = manager_enc.save()
				save_dec_path = manager_dec.save()
				print("Saved checkpoint for step {}: {}".format(int(ckpt_enc.step), save_enc_path))
				print("Saved checkpoint for step {}: {}".format(int(ckpt_dec.step), save_dec_path))

if __name__ == '__main__':
	train()