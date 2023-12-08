import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, History, CSVLogger
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras.layers import (Conv2D,AvgPool2D,Input)
from keras.activations import sigmoid
from keras.layers import LeakyReLU, PReLU


def se_block(input_feature, ration = 4):
  channel = input_feature.shape[-1]

  x =  tf.keras.layers.GlobalAveragePooling2D()(input_feature)
  x = x[:,None,None,:]

  x = tf.keras.layers.Conv2D(filters=channel//ration, kernel_size=1, strides=1)(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Conv2D(filters = channel, kernel_size = 1, strides=1)(x)
  x = tf.keras.layers.Activation('sigmoid')(x)
  x = tf.keras.layers.Multiply()([input_feature,x])
 

  return x

def CBAM(input_feature, ratio , block, partial_name):
	name_base = 'Block' + str(block)

	channel = input_feature.shape[-1]
	# channel attention
	avg_pool = GlobalAveragePooling2D(name=name_base + '_averagepooling_' + partial_name)(input_feature)
	avg_pool = Dense(channel // ratio, activation='relu', name=name_base + '_dense1_' + partial_name)(avg_pool)

  ##這裡本來是average
	max_pool = GlobalMaxPooling2D(name=name_base + '_maxpooling_' + partial_name)(input_feature)
	max_pool = Dense(channel // ratio, activation='relu',name=name_base + '_dense2_' + partial_name)(max_pool)

	avg_pool = Dense(channel, activation=None, name=name_base + '_dense3_' + partial_name)(avg_pool)	
	max_pool = Dense(channel, activation=None, name=name_base + '_dense4_' + partial_name)(max_pool)
	mask = Add(name=name_base + '_add2_' + partial_name)([avg_pool, max_pool])
	mask = Activation('sigmoid',name=name_base + '_CBAM_sigmoid_' + partial_name)(mask)
	x = multiply([input_feature,mask], name=name_base + '_multi_' + partial_name) 
	

	# spatial attention
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(x)
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(x)
	concat = Concatenate(axis=3)([avg_pool, max_pool])

	# x = Conv2D(8, (1, 1), padding='same', activation='tanh')(x)
	mask = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name=name_base + '_conv3_' + partial_name)(x)
	output = multiply([mask, x], name=name_base + '_multi2_' + partial_name)
	return output



from keras.activations import elu

def Residual_block(input_tensor, block, scale, partial_name="mix"):
    
    name_base = 'Block' + str(block)
 
    if (partial_name == "b"):
      x = tf.keras.layers.Activation('sigmoid', name=name_base + '_pre_sigmoid_' + partial_name)(input_tensor)
    else:
      x = tf.keras.layers.Activation('relu', name=name_base + '_pre_relu_' + partial_name)(input_tensor)
    
    # drop out
    x = tf.keras.layers.Dropout(0.3, name=name_base + '_pre_dropout_'+ partial_name ,noise_shape=None, seed=None)(x)
    
    # conv 1
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name=name_base + '_conv1_' + partial_name)(x)
    
    # # Activation (ELU)
    # custom_elu = lambda x: elu(x, alpha=1.0)
    # x = tf.keras.layers.Activation(custom_elu, name=name_base + '_ELU_' + partial_name)(x)
    
    
    
    if (partial_name == "b"):
      x = tf.keras.layers.Activation('sigmoid', name=name_base + '_sigmoid_' + partial_name)(x)
    else:
	    x = tf.keras.layers.Activation('relu', name=name_base + '_relu_' + partial_name)(x)
    
    # drop out
    x = tf.keras.layers.Dropout(0.3, name=name_base + '_dropout_'+ partial_name ,noise_shape=None, seed=None)(x)
    
    
    # conv 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name=name_base + '_conv2_' + partial_name)(x)
    
    x = CBAM(x,4,block,partial_name)

    x = tf.keras.layers.Lambda(lambda inputs, scale: inputs[0]+ inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale}, name=name_base + '_add_' + partial_name)([input_tensor,x])
    
    
    return x

def Residual_block_branch_pt(input_tensor, block, scale, partial_name="mix"):
    
    name_base = 'Block' + str(block)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name=name_base + '_conv1_' + partial_name)(input_tensor)
  
    if (partial_name == "b"):
	    x = tf.keras.layers.Activation('sigmoid', name=name_base + '_sigmoid_' + partial_name)(x)
    else:
	    x = tf.keras.layers.Activation('relu', name=name_base + '_relu_' + partial_name)(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name=name_base + '_conv2_' + partial_name)(x)

    input_tensor = tf.keras.layers.Conv2D(64, (1, 1), padding='same', name=name_base + '_pointwise_conv_' + partial_name)(input_tensor)
        
    x = tf.keras.layers.Lambda(lambda inputs, scale: inputs[0]+ inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale}, name=name_base + '_add_' + partial_name)([input_tensor,x])
    return x

def pgt_block_48_v26(output_branch=1, block_num=48, total_stages=24):
	input_tensor = tf.keras.layers.Input(shape=(None, None, 1), name="input")
	non_binary_input = tf.expand_dims(input_tensor[:,:,:,0], -1)

	BFM = tf.keras.layers.Conv2D(128, 3, dilation_rate=3, padding='same', name='BFM_dilated1')(non_binary_input)
	BFM = tf.keras.layers.Conv2D(256, 3, dilation_rate=2, padding='same', name='BFM_dilated2')(BFM)

	####################################################
	####### process shared features 1~24 共24
	####################################################
	for block in range(1, block_num//2 + 1):
		scale = total_stages*0.01-0.01*block
		if(scale <= 0):
			scale = scale - 0.01
		
		if block == 1:
			x = Residual_block_branch_pt(BFM, block, scale)
		else:
			x = Residual_block(x, block, scale)


	####################################################
	####### process binary features 25~60 共35
	####################################################
	for block in range(block_num//2 + 1, block_num+1 + 12):
		scale = total_stages*0.01-0.01*block
		if(scale <= 0):
			scale = scale - 0.01

		if block == (block_num//2 + 1):
			b_x = Residual_block_branch_pt(x, block, scale, "b")
		else:
			b_x = Residual_block(b_x, block, scale, "b")
	
	b_x = tf.keras.layers.Concatenate(name='Superimpose_b')([b_x, BFM])
	b_x = tf.keras.layers.Conv2D(1, (1, 1), padding='same', name='final_conv_b')(b_x)


	####################################################
	####### process non-binary features 25~48 共24
	####################################################
	x_bx = tf.keras.layers.Concatenate(name='Superimpose_xb')([x, b_x])
	for block in range(block_num//2 + 1, block_num//2 + 1 + 24):
		scale = total_stages*0.01-0.01*block
		if(scale <= 0):
			scale = scale - 0.01

		if block == (block_num//2 + 1):
			nb_x = Residual_block_branch_pt(x_bx, block, scale, "nb")
		else:
			nb_x = Residual_block(nb_x, block, scale, "nb")


	nb_x = tf.keras.layers.Concatenate(name='Superimpose_nb')([nb_x, BFM])
	nb_x = tf.keras.layers.Conv2D(1, (1, 1), padding='same', name='final_conv_nb')(nb_x)
	
	x = tf.keras.layers.Concatenate(name='final_concatenate')([nb_x,b_x])
	# model = tf.keras.models.Model(inputs=input_tensor , outputs = x)
  
	#ori_x = tf.keras.layers.Conv2D(36, (12, 12), strides=12, padding='same', name='ori_nb')(nb_x)
	#ori_x = tf.keras.layers.Dense(36,activation="softmax",use_bias=True,name="ori_conv_nb")(ori_x)


  
	# pickout_x = tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding='same', name='pickout_conv_nb')(nb_x)
	# # pickout_x = tf.keras.activations.elu(pickout_x, alpha=1.0)
	# # pickout_x = tf.keras.layers.Conv2D(64, 3)(pickout_x)
	# pickout_x = tf.keras.layers.Dense(36*36,activation="sigmoid",use_bias=True,name="pickout_dense_nb")(pickout_x)

	# pickout_x = tf.keras.layers.GlobalAveragePooling2D(name="pickout_g_pooling_nb")(pickout_x)

	model = tf.keras.models.Model(inputs=input_tensor , outputs = x)

	
	return model
'''
	####################################################
	####### process binary features
	####################################################
	for block in range(block_num//2 + 1, block_num+1 + 12):
		scale = total_stages*0.01-0.01*block
		if(scale <= 0):
			scale = scale - 0.01

		if block == (block_num//2 + 1):
			b_x = Residual_block_branch_pt(x, block, scale, "b")
		else:
			b_x = Residual_block(b_x, block, scale, "b")
	
	b_x = tf.keras.layers.Concatenate(name='Superimpose_b')([b_x, BFM])
	b_x = tf.keras.layers.Conv2D(1, (1, 1), padding='same', name='final_conv_b')(b_x)
	
'''

'''	if output_branch == 1:
		model = tf.keras.models.Model(inputs=input_tensor, outputs = x)
	else:
		model = tf.keras.models.Model(inputs=input_tensor, outputs = [x,x])
'''

if __name__ == "__main__":
	model = pgt_block_48_v26()
	
	model.summary()
	tf.keras.utils.plot_model(model,to_file='/content/model.jpg',show_shapes=True)