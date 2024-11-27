from PIX2PIX.BuildModel import *
from PIX2PIX import DataProcessing

# Input dataset
Input_FolderList = [r'dataset/Input/1',r'dataset/Input/2',r'dataset/Input/3',r'dataset/Input/4'] # Replace your several single-channel folder
InputData_path = r'dataset/Input/Concate'
inputData = DataProcessing.DataChannelConvert(Input_FolderList, InputData_path)
inputData.final_process_multi_channels_img()
Input_channels = inputData.chan
Input_numbers = inputData.imgNum

# Target dataset
Target_FolderList = [r'dataset/Target/5'] # Replace your several single-channel folder
TargetData_path = r'dataset/Target/Concate'
TargetData = DataProcessing.DataChannelConvert(Target_FolderList, TargetData_path)
TargetData.final_process_multi_channels_img()
Target_channels = TargetData.chan
Target_numbers = TargetData.imgNum


# Initialize model
generator = Generator(Input_channels,Target_channels)
discriminator = Discriminator(Input_channels,Target_channels)

# Optimization
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Checkpoint setting
checkpoint_dir = 'model/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Define parameters
BUFFER_SIZE = Input_numbers
BATCH_SIZE = 1
EPOCHS = 500

# Create official data sets
train_dataset = load_dataset(InputData_path, TargetData_path)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# !!!!!!!!!!!!!!!!!!!! Train !!!!!!!!!!!!!!!!!!!!!!
if __name__ == "__main__":
    fit(train_dataset, EPOCHS, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint, checkpoint_prefix)

