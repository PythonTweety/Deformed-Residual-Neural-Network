from tensorflow import keras 
from tensorflow.keras.callbacks import ModelCheckpoint, History, CSVLogger
import loaddata, total_loss

from pgt_block_48_v26 import pgt_block_48_v26

import numpy as np
import cv2
from PIL import Image
import os
from tensorflow.python.client import device_lib

import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)
print("tf.config.experimental.list_physical_devices('GPU') = ", tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


############################ self-defined testing variables ############################
nb_test_img_path = '/content/drive/MyDrive/DRBFD_20230904/testset_nasic9395_v9/identify'										# non-binary testset input
b_test_img_path = '/content/drive/MyDrive/DRBFD_20230904/testset_nasic9395_v9/identify'									# path of binary testing output

nb_normal_img_path = '/content/drive/MyDrive/DRBFD_20230904/testset_nasic9395_v9/identify'
b_normal_img_path = '/content/drive/MyDrive/DRBFD_20230904/testset_nasic9395_v9/identify_b'

b_test_result_path = '/content/drive/MyDrive/DRBFD_20230904/testset_nasic9395_v9/de_identify_b'		
nb_test_result_path ='/content/drive/MyDrive/DRBFD_20230904/testset_nasic9395_v9/de_identify'
############################ self-defined other variables ############################
lr = 0.001
epochs = 50
batch_size = 15
load_weights = '' # put testing model here
training = False
testing = True

print("model=", load_weights)


def scheduler(epoch, lr):
	if(epoch == 0):
		with open(lr_csv_path, "w") as f:
			f.write("epoch,lr\n")
			if epoch < 25:
				print("epoch = %d, lr = %f" %(epoch, lr))
				f.write("epoch = %d, lr = %f" %(epoch, lr))
				return lr
			else:
				print("epoch = %d, lr = %f" %(epoch, lr * 0.9))
				f.write("epoch = %d, lr = %f" %(epoch, lr * 0.9))
				return lr * 0.9
	else:
		with open(lr_csv_path, "a") as f:
			if epoch < 25:
				print("epoch = %d, lr = %f" %(epoch, lr))
				f.write("epoch = %d, lr = %f" %(epoch, lr))
				return lr
			else:
				print("epoch = %d, lr = %f" %(epoch, lr * 0.9))
				f.write("epoch = %d, lr = %f" %(epoch, lr * 0.9))
				return lr * 0.9



def main():
	model = pgt_block_48_v26()
	
	
	if load_weights!='':
		model.load_weights(load_weights)
	
	if testing:
		test(model=model)





def test(model=None):
	print('============================')
	print('start testing...')
	print(len(os.listdir(nb_test_img_path)))

	counter = 0.0
	nb_total_MSE_error = 0.0
	nb_total_SSIM_error = 0.0
	nb_total_PSNR_error = 0.0
	b_total_MSE_error = 0.0
	b_total_SSIM_error = 0.0
	b_total_PSNR_error = 0.0

	for img_file in os.listdir(nb_test_img_path):
		print(img_file)
		if(img_file != "samp"):
			print(img_file)
			nb_img = cv2.imread(nb_test_img_path + '/' + img_file, cv2.IMREAD_GRAYSCALE) / 255.0
			nb_img = nb_img.reshape((1, nb_img.shape[0], nb_img.shape[1], 1))
			append_img = nb_img
			
			nb_img_normal = cv2.imread(nb_normal_img_path + "/" + img_file, cv2.IMREAD_GRAYSCALE) / 255.0
			b_img_normal = cv2.imread(b_normal_img_path + "/" + img_file, cv2.IMREAD_GRAYSCALE) / 255.0
			
			
			figerprint_enhanced_test = model.predict([append_img])
			all_img_result = np.squeeze(figerprint_enhanced_test).astype(np.float32)
    
			figerprint_enhanced_test = model.predict([append_img])
			all_img_result = np.squeeze(figerprint_enhanced_test).astype(np.float32)
			

			print("all_img_result.shape = ", all_img_result.shape)
			print("len(all_img_result.shape) = ", len(all_img_result.shape))
			if(len(all_img_result.shape) == 2):
				print(img_file, " error")
				continue
			nb_img_result = all_img_result[0:all_img_result.shape[0], 0:all_img_result.shape[1], 0]
			print("nb_img_result.shape = ", nb_img_result.shape)
			b_img_result = all_img_result[0:all_img_result.shape[0], 0:all_img_result.shape[1], 1]

			nb_mse_error = mse(nb_img_result, nb_img_normal)
			nb_ssim_error = ssim(nb_img_result, nb_img_normal)
			nb_img_result = (np.clip(nb_img_result*255,0,255)).astype(np.uint8)
			nb_img_normal = (np.clip(nb_img_normal*255,0,255)).astype(np.uint8)
			nb_psnr_error = psnr(nb_img_result, nb_img_normal)
			print("nb_img_result")
			print("MSE = ", round(nb_mse_error, 3))
			print("SSIM = ", round(nb_ssim_error, 3))
			print("PSNR = ", round(nb_psnr_error, 3))


			b_mse_error = mse(b_img_result, b_img_normal)
			b_ssim_error = ssim(b_img_result, b_img_normal)
			b_img_result = (np.clip(b_img_result*255,0,255)).astype(np.uint8)
			b_img_normal = (np.clip(b_img_normal*255,0,255)).astype(np.uint8)
			b_psnr_error = psnr(b_img_result, b_img_normal)
			print("b_img_result")
			print("MSE = ", round(b_mse_error, 3))
			print("SSIM = ", round(b_ssim_error, 3))
			print("PSNR = ", round(b_psnr_error, 3))

			cv2.imwrite(nb_test_result_path + "/" + img_file, nb_img_result)
			cv2.imwrite(b_test_result_path + "/" + img_file, b_img_result)
			
			counter += 1
			nb_total_MSE_error += nb_mse_error
			nb_total_SSIM_error += nb_ssim_error
			nb_total_PSNR_error += nb_psnr_error

			b_total_MSE_error += b_mse_error
			b_total_SSIM_error += b_ssim_error
			b_total_PSNR_error += b_psnr_error


		
	
	print("average nb_MSE error over %d images = %f" %(counter, nb_total_MSE_error/counter))
	print("average nb_SSIM error over %d images = %f" %(counter, nb_total_SSIM_error/counter))
	print("average nb_PSNR error over %d images = %f" %(counter, nb_total_PSNR_error/counter))
	print("average b_MSE error over %d images = %f" %(counter, b_total_MSE_error/counter))
	print("average b_SSIM error over %d images = %f" %(counter, b_total_SSIM_error/counter))
	print("average b_PSNR error over %d images = %f" %(counter, b_total_PSNR_error/counter))

if __name__ == "__main__":
	
	main()
	

	print('*****************************')
	print('finish!')
