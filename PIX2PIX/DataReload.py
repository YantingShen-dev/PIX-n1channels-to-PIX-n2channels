import os
import numpy as np
import tifffile as tiff
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Save temporary image using tifffile
def save_temp_image(input_image_path, temp_folder, image_name):
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    temp_image_path = os.path.join(temp_folder, image_name)
    # Load the input image using tifffile
    input_image = tiff.imread(input_image_path)
    # Save the image using tifffile
    tiff.imwrite(temp_image_path, input_image)


# Predict target output from the model
def predict_target(model, input):
    prediction = model(input, training=True)
    prediction_np = prediction[0].numpy()
    # [-1, 1] --> [0, 255]
    prediction_np = ((prediction_np + 1) * 127.5).astype(np.uint8)
    return prediction_np


# Predict a folder of images using the model and save them as TIFF
def predict_folder(model, dataset, file_name, output_folder=r'prediction', num_images=0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if num_images == 0:
        num_images = sum(1 for _ in dataset)

    for input_image, target in dataset.take(num_images):
        # Get the prediction from the model
        prediction_np = predict_target(model, input_image)
        prediction_np = prediction_np.transpose((2,0,1))

        # Save the prediction to the output folder as TIFF
        output_path = os.path.join(output_folder, file_name)
        tiff.imwrite(output_path, prediction_np)  # Use tifffile to save as TIFF

        print(f"{file_name} saved.")


# Calculate MAE and MSE error metrics and display
def calculate_error_metrics_and_display(model, dataset):
    for input_image, target in dataset:
        # Get prediction from the model
        prediction = model(input_image, training=True)

        # Convert EagerTensors to numpy arrays and flatten
        target_np = target.numpy().flatten()
        prediction_np = prediction.numpy().flatten()

        # Calculate MAE and MSE
        mae = mean_absolute_error(target_np, prediction_np)
        mse = mean_squared_error(target_np, prediction_np)

    return mae, mse
