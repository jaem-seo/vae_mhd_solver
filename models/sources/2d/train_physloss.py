#!/usr/bin/env python

import os,sys
import numpy as np
import tensorflow as tf
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,CSVLogger

try:
    i_seed = int(sys.argv[1])
    pwd_ = sys.argv[2]
except:
    i_seed = 0
    pwd_ = '.'
split = 0.8
file_save = pwd_+'/best_model{}'.format(i_seed)
file_check = pwd_+'/checkpoint{}'.format(i_seed)
np.random.seed(i_seed)
tf.random.set_seed(i_seed)
epsilon = 1.e-8
beta = 0.01

data_path = '../../../input_gen/data/total_data.npz'

def r2(y_true, y_pred):
    SS_res = np.sum(np.square(y_true - y_pred))
    SS_tot = np.sum(np.square(y_true - y_true.mean(axis=0)))
    return ( 1 - SS_res/(SS_tot + epsilon) )

def r2_k(y_true, y_pred):
    SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + tf.keras.backend.epsilon()) )

def grad(v):
    #vvstack = tf.experimental.numpy.vstack((v[[0]],v))
    #vhstack = tf.experimental.numpy.hstack((v[:,[0]],v))
    #vvstack = tf.keras.layers.Concatenate(axis=1)([v[:,[0]],v])
    #vhstack = tf.keras.layers.Concatenate(axis=2)([v[:,:,[0]],v])
    vvstack = tf.keras.layers.Concatenate(axis=1)([tf.reshape(v[:,0],[-1,1,129]),v])
    vhstack = tf.keras.layers.Concatenate(axis=2)([tf.reshape(v[:,:,0],[-1,129,1]),v])
    #return tf.experimental.numpy.diff(vvstack,axis=1), tf.experimental.numpy.diff(vhstack,axis=2)
    return vvstack[:,1:]-vvstack[:,:-1], vhstack[:,:,1:]-vhstack[:,:,:-1]

def physloss(psi,phi):
    psix,psiy = grad(psi)
    phix,phiy = grad(phi)
    return tf.reduce_mean(tf.reduce_sum(tf.square(psix*phiy-psiy*phix),axis=(1,2)))

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="rec_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.phys_loss_tracker = keras.metrics.Mean(name="phys_loss")

    def call(self,x):
        z_mean, z_log_var, z = self.encoder(x)
        y = self.decoder(z_mean)
        return y

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.phys_loss_tracker
        ]

    def train_step(self, data):
        x_data, y_data = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x_data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(y_data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            phys_loss = physloss(reconstruction[:,:,:,0],reconstruction[:,:,:,1])
            total_loss = reconstruction_loss + beta*kl_loss + phys_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.phys_loss_tracker.update_state(phys_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "phys_loss": self.phys_loss_tracker.result()
        }

    def test_step(self, data):
        x_data, y_data = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x_data)
            reconstruction = self.decoder(z_mean)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(y_data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            phys_loss = physloss(reconstruction[:,:,:,0],reconstruction[:,:,:,1])
            total_loss = reconstruction_loss + beta*kl_loss + phys_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.phys_loss_tracker.update_state(phys_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "phys_loss": self.phys_loss_tracker.result()
        }


t_start = time.time()

# Preprocess
print('Preprocessing...')
#data_tmp = np.load('../../data/total_data.npz')
data_tmp = np.load(data_path)
data_in = np.zeros([len(data_tmp['aspect']),201,6])
data_in[:,:,0] = np.repeat([data_tmp['aspect'].T],201,axis=0).T
data_in[:,:,1] = np.repeat([data_tmp['kappa'].T],201,axis=0).T
data_in[:,:,2] = np.repeat([data_tmp['deltau'].T],201,axis=0).T
data_in[:,:,3] = np.repeat([data_tmp['deltal'].T],201,axis=0).T
data_in[:,:,4] = data_tmp['j_star']
data_in[:,:,5] = data_tmp['p_star']
psi_star = np.reshape(data_tmp['psi_star'],(-1,129,129,1))
phi_star = np.reshape(data_tmp['phi_star'],(-1,129,129,1))
data_out = np.append(psi_star,phi_star,axis=-1)
len_train = int(split*len(data_in))
train_data_in = data_in[:len_train]
train_data_out = data_out[:len_train]
validation_data_in = data_in[len_train:]
validation_data_out = data_out[len_train:]
del data_in,data_out,data_tmp,psi_star,phi_star
print('Preprocessed.')
print('train_data_in shape:',np.shape(train_data_in))
print('train_data_out shape:',np.shape(train_data_out))
print('Elapsed time:', time.time()-t_start)

# Build model
print('Building model...')

# Encoder
latent_dim = 16
encoder_inputs = keras.Input(shape=(201, 6))
x = layers.Conv1D(64, 3, activation="relu", padding="same")(encoder_inputs)
x = layers.MaxPooling1D(3,padding='same')(x)
x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling1D(3,padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(32 * 32 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((32, 32, 64))(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x) # 64 * 64 * 128
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2)(x) # 129 * 129 * 64
decoder_outputs = layers.Conv2DTranspose(2, 3, padding="same")(x) # 129 * 129 * 2
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# Train the VAE
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
callbacks = [EarlyStopping(monitor='val_rec_loss',patience=500,restore_best_weights=True),
            ModelCheckpoint(filepath=file_check,monitor='val_loss',save_best_only=True),
            CSVLogger('training{}.log.csv'.format(i_seed),append=True)]
vae.fit(train_data_in, train_data_out, epochs=2000, batch_size=128, callbacks=callbacks, validation_data=(validation_data_in,validation_data_out))#, verbose=2)
vae.encoder.save(file_save+'_enc')
vae.decoder.save(file_save+'_dec')

