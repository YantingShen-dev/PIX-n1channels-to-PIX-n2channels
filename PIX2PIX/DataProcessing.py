import os
import tensorflow as tf
import tifffile as tiff


class DataChannelConvert(object):
    switch = True

    def __init__(self, folder_list, output_folder):
        self.chan = len(folder_list)
        self.folder_list = folder_list
        self.output_folder = output_folder
        self.imgNum = self.count_files(folder_list[0])
        if os.path.exists(output_folder):
            self.switch = False

    def count_files(self, folder_path):
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return len(files)

    # Save images to specific path
    def save_image(self, image, file_name):
        # [-1, 1] --> [0, 255]
        image = (image + 1) * 127.5
        image = tf.cast(image, tf.uint8)
        image = image.numpy()
        image = image.transpose((2,0,1))  # (256, 256, 5) -> (5, 256, 256)

        # Save as TIFF
        base_name = os.path.splitext(file_name)[0]  # Removes any existing extension
        output_path = os.path.join(self.output_folder, base_name + '.tiff')
        tiff.imwrite(output_path, image)

    # Load single channel image (1 channel, grayscale)
    def load_single_channel_image(self, path):
        if self.switch:
            image = tf.io.read_file(path)
            image = tf.image.decode_png(image, channels=1)  # Decode as single-channel grayscale
            image = tf.image.resize(image, [256, 256])  # Resize to (256, 256)
            image = (image / 127.5) - 1  # Normalize to [-1, 1]
            return image

    # Load and process multi-channel image
    def create_multi_channels_img(self, file_name):
        if self.switch:
            all_img = []
            for sub_folder in self.folder_list:
                img_path = os.path.join(sub_folder, file_name)
                img = self.load_single_channel_image(img_path)
                if img is not None:
                    all_img.append(img)  # Append each channel as a separate image
            # Concatenate images along the last axis (channel axis)
            multi_channels_img = tf.concat(all_img, axis=-1)
            return multi_channels_img

    # Final processing for multi-channel images
    def final_process_multi_channels_img(self):
        if self.switch:
            # Get all file names from the first folder
            file_names = [f for f in os.listdir(self.folder_list[0]) if f.endswith(('.png', '.jpg', '.TIFF'))]

            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

            for file_name in file_names:
                multi_channels_img = self.create_multi_channels_img(file_name)
                if multi_channels_img is not None:
                    self.save_image(multi_channels_img, file_name)  # Save the multi-channel image as TIFF
                    print(f"Processed and saved {self.chan} channel image: {self.output_folder}")
                else:
                    print(f"Skip file {file_name}: One or more single-channel images cannot be loaded.")


if __name__ == "__main__":
    folder_li = [r'../dataset/Input/4channels/1', r'../dataset/Input/4channels/2', r'../dataset/Input/4channels/3',
                 r'../dataset/Input/4channels/4', r'../dataset/Input/4channels/4']
    out_folder = r'../dataset/Input/4channels/After'
    inputData = DataChannelConvert(folder_li, out_folder)
    inputData.final_process_multi_channels_img()
    print(inputData.chan)
