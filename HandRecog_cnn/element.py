from PIL import Image
import numpy as np

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((96, 96))
    image_np = np.array(image, dtype=np.float32)
    image_np = (image_np / 255.0 - 0.0979) / 0.1991
    input_exponent = -13
    image_np = np.clip(image_np * (1 << -input_exponent), -32768, 32767).astype(np.int16)


    print("{", end="")
    for i, pixel in enumerate(image_np.flatten()):
        print(f"{pixel}", end=", " if i < len(image_np.flatten()) - 1 else "")
    print("}")

preprocess_image(r'D:\Projects\HandRecog_cnn\dataset\test\08\07_ok\frame_08_07_0159.png')
