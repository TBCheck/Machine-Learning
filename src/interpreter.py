import os
from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np

model_path = "model.tflite"
test_img_path = "D://document//target//2021-2022//event//bootcamp//bangkit//capstone//Presentasi//ML//ML-task/data/testing"

labels = ['Normal', 'Tuberculosis']

#####   tflite interpreter setup    #####
#####-------------------------------#####
#####-------------------------------#####
#####-------------------------------#####
#####-------------------------------#####

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


#####   Image convert & preprocess  #####
#####-------------------------------#####
#####-------------------------------#####
#####-------------------------------#####
#####-------------------------------#####

# Get image files
normal_img = []
tb_img = []
for img_dir in os.listdir(test_img_path):
    # print(img_dir)
    if img_dir == "normal":
        normal_path = os.path.join(test_img_path, img_dir)
        for img in os.listdir(normal_path):
            normal_img.append(os.path.join(normal_path, img))

    if img_dir == "tuberculosis":
        tb_path = os.path.join(test_img_path, img_dir)
        for img in os.listdir(tb_path):
            tb_img.append(os.path.join(tb_path, img))

# print(tb_img[0])

# Read image with Pillow
# Only one image
# img = Image.open(normal_img[0]).convert('RGB')
img = Image.open(tb_img[0]).convert('RGB')

# Get input size
input_shape = input_details[0]['shape']
size = input_shape[:2] if len(input_shape) == 3 else input_shape[1:3]

# Preprocess image
img = img.resize(size)
img = np.array(img, dtype=np.float32)
img = img / 255.

# Add a batch dimension
input_data = np.expand_dims(img, axis=0)


#####   Run prediction using tflite  #####
#####--------------------------------#####
#####--------------------------------#####
#####--------------------------------#####
#####--------------------------------#####

# Point the data to be used for testing and run the interpreter
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Obtain results and print the predicted category
predictions = interpreter.get_tensor(output_details[0]['index'])
predicted_label = np.array(predictions).flatten()
predicted_label = predicted_label[0]
if predicted_label >= 0.8:
    print(predicted_label, ": ", labels[1])
else:
    print(predicted_label, ": ", labels[0])