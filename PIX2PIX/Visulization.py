from PIL import Image, ImageDraw, ImageFont
import numpy as np

def generate_images(model, input_image, target):
    prediction = model(input_image, training=True)

    input_image_np = input_image[0].numpy()
    target_image_np = target[0].numpy()
    prediction_np = prediction[0].numpy()

    # [-1, 1] --> [0, 255]
    input_image_np = ((input_image_np + 1) * 127.5).astype(np.uint8)
    target_image_np = ((target_image_np + 1) * 127.5).astype(np.uint8)
    prediction_np = ((prediction_np + 1) * 127.5).astype(np.uint8)

    input_image_np = input_image_np[..., :4]
    target_image_np = target_image_np[..., :4]
    prediction_np = prediction_np[..., :4]

    # Delete input_img batch
    input_image_np = np.squeeze(input_image_np, axis=-1) if input_image_np.shape[-1] == 1 else input_image_np
    target_image_np = np.squeeze(target_image_np, axis=-1) if target_image_np.shape[-1] == 1 else target_image_np
    prediction_np = np.squeeze(prediction_np, axis=-1) if prediction_np.shape[-1] == 1 else prediction_np

    # Convert NumPy arrays to PIL images
    input_image_pil = Image.fromarray(input_image_np)
    target_image_pil = Image.fromarray(target_image_np)
    prediction_pil = Image.fromarray(prediction_np)

    # Get the width and height of the image
    width, height = input_image_pil.size

    # Create a new empty image for combining the three images and reserve space for a caption
    combined_image = Image.new('RGBA', (width * 3, height + 40), (255, 255, 255))  # 40 pixels for titles

    # Add Title
    title_font = ImageFont.load_default()
    draw = ImageDraw.Draw(combined_image)

    # Paste the three images sequentially into the combined image
    combined_image.paste(input_image_pil.convert("RGBA"), (0, 40))  # Input
    combined_image.paste(target_image_pil.convert("RGBA"), (width, 40))  # Ground truth
    combined_image.paste(prediction_pil.convert("RGBA"), (width * 2, 40))  # prediction

    # Add a title above
    draw.text((width // 2 - 30, 10), "Input", font=title_font, fill=(0, 0, 0))
    draw.text((width + width // 2 - 50, 10), "Ground truth", font=title_font, fill=(0, 0, 0))
    draw.text((2 * width + width // 2 - 50, 10), "prediction", font=title_font, fill=(0, 0, 0))

    return combined_image