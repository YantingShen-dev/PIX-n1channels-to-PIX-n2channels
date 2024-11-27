## PIX(n1_channel) to PIX(n2_channel)

There are more details in my [paper](https:xxx):

> **- Reference -**  
> pix2pix: https://github.com/phillipi/pix2pix  
> pix2pix for tensorflow: https://github.com/affinelayer/pix2pix-tensorflow


### Description

1) This model creates a mapping from （n-channel） images to （n-channel） images
2) Follows the traditional pix2pix modeling framework, which automatically scales the image size to 256*256

   <img src="docs\1.png" width="900px"/>  

3) Since the image display contains a maximum of four RGBA channels, the visualization cannot display all channel information for images with more than four channels.


4) Set up dependencies: 

    ```shell
    pip install -r requirements.txt
    ```

   The code was tested with Python 3.7, tensorflow 2.9.0

### Usage

1) **DataPreparation:** Images of channels have the same filename and be placed in separate subfolders in `dataset/Input`.( ! ! Must be a single channel)

2) **ModelTraing:** Replace _'Input_FolderList'_ and _'Target_FolderList'_ in `train.py` with _'Input_Image'_ and _'Target_Image'_.

    ```shell
    Input_FolderList = [r'dataset/Input/1',r'dataset/Input/2',...,r'dataset/Input/n1']
    Target_FolderList = [r'dataset/Target/1',r'dataset/Target/2',...,r'dataset/Input/n2']
    ```

    _'InputData_path'_ and _'TargetData_path'_ contains the Input and Target after multi-channels stacking.

    ```shell
    InputData_path = r'dataset/Input/Concate'
    TargetData_path = r'dataset/Target/Concate'
    ```
   
   You can check the training procedure in `procedure` folder.  
   Model is saved every 500 epochs in `model` folder.
   <img src="docs\2.png" width="500px"/> 


3) **PredictImage:** Replace _'InputData_path'_ and _'TargetData_path'_ in `predict.py` with the image and target to be predicted for MAE and MSE.(predicted image saved in _'output_folder'_)

   ```shell
    InputData_path = r'dataset/Input/Concate'
    TargetData_path = r'dataset/Target/Concate'
    output_folder = r'prediction'
    ```


### Additional notes
A simple three-channel-to-three-channel pix2pix model was slightly modified to implement a multi-channel-to-multi-channel mapping.  
The main application is reflected in the fact that different channels contain information about different variables, by which the multivariate-to-multivariate mapping modeling can be realized, and then implicit relationships can be uncovered from pixel convolution.
