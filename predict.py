import sys
import cv2
import numpy as np
import tensorflow as tf

if len(sys.argv) == 2:
    try:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 512))  # Ensure it matches the input size
        img = img / 255.0  # Normalize to [0, 1]
        img = np.expand_dims(img, axis=-1)  # Shape becomes (512, 512, 1)
        img = np.expand_dims(img, axis=0)   # Shape becomes (1, 512, 512, 1)

        model = tf.keras.models.load_model('model.h5')

        prediction = model.predict(img)

        # 0 indicates Malignant
        # 1 indicates Normal
        chance = prediction[0, 0]

        if chance < 0.1:
            print("This patient likely has lung cancer")
        elif chance > 0.9:
            print("This patient likely does NOT have lung cancer")
        else:
            print("Unsure whether this patient has lung cancer or not")
        print(f"{100*(1-chance):.3f}% chance of having lung cancer")
        print(f"{100*(chance):.3f}% chance of NOT having lung cancer")
    except:
        print("Invalid argument")
elif len(sys.argv) < 2:
    print("Image required")
else:
    print("Too many arguments")