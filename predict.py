from PIX2PIX.DataReload import *
from train import *

# Reload pre-trained model
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Select image folder to be predicted and analysed
InputData_path =  r'dataset/Input/Concate'
TargetData_path = r'dataset/Target/Concate'
output_folder = r'prediction'

image_files = []
total_mae = 0
total_mse = 0
sample_num = len(os.listdir(InputData_path))

for filename in os.listdir(InputData_path):
    new_InputData_path = os.path.join(InputData_path,filename)
    new_TargetData_path = os.path.join(TargetData_path,filename)

    temp_InputData_path = r'PIX2PIX/Temp/Input'
    temp_TargetData_path = r'PIX2PIX/Temp/Target'

    save_temp_image(new_InputData_path, temp_InputData_path, filename)
    save_temp_image(new_TargetData_path, temp_TargetData_path, filename)

    # Predict and save corresponding input_img
    dataset = load_dataset(temp_InputData_path, temp_TargetData_path)
    dataset = dataset.batch(BATCH_SIZE)
    predict_folder(generator, dataset,filename, output_folder)

    # MAE & MSE
    mae,mse = calculate_error_metrics_and_display(generator, dataset)
    total_mae += mae
    total_mse += mse
    print(f"{filename} --->MAE:{mae},MSE:{mse}")

    # Delete
    os.remove(os.path.join(temp_InputData_path, filename))
    os.remove(os.path.join(temp_TargetData_path, filename))

print(f"Total MAE:{total_mae/sample_num} MSE{total_mse/sample_num}")
