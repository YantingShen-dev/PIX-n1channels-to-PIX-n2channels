import tensorflow as tf
from tensorflow.keras import layers
import os
import tifffile as tiff
import time
from PIX2PIX.Visulization import generate_images

# Generator and Discriminator
# Down Sampling Module
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(filters, size, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result

# Upper Sampling Module
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(filters, size, strides=2,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False))
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

# Generator (U-Net) Modify input to (256, 256, n_input) and output to (256, 256, n_target)
def Generator(n_input, n_target):
    inputs = layers.Input(shape=[256, 256, n_input])

    # Downsampling layers (Encoder)
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (128, 128, 64)
        downsample(128, 4),  # (64, 64, 128)
        downsample(256, 4),  # (32, 32, 256)
        downsample(512, 4),  # (16, 16, 512)
        downsample(512, 4),  # (8, 8, 512)
        downsample(512, 4),  # (4, 4, 512)
        downsample(512, 4),  # (2, 2, 512)
        downsample(512, 4),  # (1, 1, 512)
    ]
    # Upsampling layers (Decoder)
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (8, 8, 1024)
        upsample(512, 4),  # (16, 16, 1024)
        upsample(256, 4),  # (32, 32, 512)
        upsample(128, 4),  # (64, 64, 256)
        upsample(64, 4),  # (128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0., 0.02)

    # Last layer: output with n_target channels
    # For multi-channels output, use n_target channels instead of 1.
    # Activation function can be 'tanh' if output is in range [-1, 1], or 'sigmoid' for [0, 1] range.
    last = layers.Conv2DTranspose(n_target, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh')  # or 'sigmoid' based on your output range

    # Build the model
    x = inputs
    skips = []

    # Encoder (Downsampling)
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Decoder (Upsampling)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


# Discriminators (PatchGAN) that require modification of input and target images
# Discriminator (PatchGAN) that supports multi-channels input and target images
def Discriminator(n_input, n_target):
    initializer = tf.random_normal_initializer(0., 0.02)

    # Input shape as n_input-channel input image (256, 256, n_input)
    inp = layers.Input(shape=[256, 256, n_input], name='input_image')

    # Target image shape as n_target-channel output image (256, 256, n_target)
    tar = layers.Input(shape=[256, 256, n_target], name='target_image')

    # Concatenate input and target images along the channel axis
    x = layers.concatenate([inp, tar])  # Shape becomes (256, 256, n_input + n_target)

    # Downsampling (Encoder) layers
    down1 = downsample(64, 4, apply_batchnorm=False)(x)  # (128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (32, 32, 256)
    down4 = downsample(512, 4)(down3)  # (16, 16, 512)

    # Add padding before the last convolution to maintain the spatial dimensions
    zero_pad1 = layers.ZeroPadding2D()(down4)  # (18, 18, 512)
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)  # (15, 15, 512)
    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)

    # Apply padding again before the final convolutional layer
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)  # (17, 17, 512)
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  # (14, 14, 1)

    # Return the model with multi-channels input and single-channel output (PatchGAN)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


# Data preprocessing: modified to load n_input channel inputs and n_output channel outputs
def load_image(input_folder, target_folder):
    input_folder = input_folder.numpy().decode('utf-8')
    filename = os.path.basename(input_folder)
    targetImg_path = os.path.join(target_folder.numpy().decode('utf-8'), filename)

    # Load input and target images using tifffile
    inputImg = tiff.imread(input_folder)
    inputImg = inputImg.transpose((1, 2, 0))
    targetImg = tiff.imread(targetImg_path)
    targetImg = targetImg.transpose((1, 2, 0))

    # Resize to 256x256 if needed (assuming the images are of shape larger than 256x256)
    inputImg = tf.image.resize(inputImg, (256, 256))
    targetImg = tf.image.resize(targetImg, (256, 256))

    # Normalize image values to range [-1, 1]
    inputImg = (inputImg / 127.5) - 1
    targetImg = (targetImg / 127.5) - 1

    # Convert to TensorFlow tensors and expand dims to match required shape (256, 256, n_input)
    inputImg_resized = tf.convert_to_tensor(inputImg, dtype=tf.float32)
    targetImg_resized = tf.convert_to_tensor(targetImg, dtype=tf.float32)

    return inputImg_resized, targetImg_resized


def load_image_wrapper(input_folder, target_folder):
    return tf.py_function(load_image, [input_folder, target_folder], [tf.float32, tf.float32])


def load_dataset(input_folder, target_folder):
    input_paths = tf.data.Dataset.list_files(os.path.join(input_folder, '*.tiff'), shuffle=False)
    dataset = input_paths.map(lambda input_path: load_image_wrapper(input_path, tf.convert_to_tensor(target_folder)))
    return dataset


def generator_loss(disc_generated_output, gen_output, target, loss_object, LAMBDA):
    # GAN Loss
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # L1 Loss
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    # Loss
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


@tf.function
def train_step(input_image, target, generator, discriminator, generator_optimizer, discriminator_optimizer):
    with tf.GradientTape(persistent=True) as tape:
        # Generate images
        gen_output = generator(input_image, training=True)

        # The discriminator calculates the truth of the real image and the generated image
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        # Calculate the loss of generators and discriminators
        gen_loss = generator_loss(disc_generated_output, gen_output, target, loss_object, LAMBDA)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # Calculate the ladder of generators and discriminators
    generator_gradients = tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply gradient update parameters
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

def fit(train_ds, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint, checkpoint_prefix,procedure_path = r'procedure/'):
    for epoch in range(epochs):
        start = time.time()
        print(f"Start {epoch+1} round of training.")

        for n, (input_image, target) in train_ds.enumerate():
            train_step(input_image, target, generator, discriminator, generator_optimizer, discriminator_optimizer)

        # Models saved every 500 rounds
        if (epoch + 1) % 500 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        # Visualize the generated image every 10 epoch
        if (epoch + 1) % 10 == 0:
            for example_input, example_target in train_ds.take(1):
                if not os.path.exists(procedure_path):
                    os.makedirs(procedure_path)
                imgtosave = generate_images(generator, example_input, example_target)
                save_path = os.path.join(procedure_path, f"{epoch + 1}.png")
                imgtosave.save(save_path)
        print(f"Time spent on {epoch+1} round of training: {time.time() - start:.2f} seconds\n")

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LAMBDA = 100
