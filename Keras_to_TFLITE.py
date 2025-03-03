import tensorflow as tf
# Load the Keras model
model = tf.keras.models.load_model("/mnt/d/Documents/ZZZ Thesis/Resnet50V2(newgen_2_22_25)50e_uf20_adam.keras")

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open("/mnt/d/Documents/ZZZ Thesis/Resnet50V2(newgen_2_22_25)50e_uf20_adam.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TFLite successfully!")# Save the TensorFlow Lite model