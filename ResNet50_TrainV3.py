import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2, ResNet50, InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset paths
train_dir = "ResNet_Dataset/train"  # Replace with your dataset path
val_dir = "ResNet_Dataset/valid"      # Replace with your dataset path

# train_datagen = ImageDataGenerator(dtype = 'float32', preprocessing_function=tf.keras.applications.resnet.preprocess_input)
# val_datagen = ImageDataGenerator(dtype = 'float32', preprocessing_function=tf.keras.applications.resnet.preprocess_input)

train_datagen = ImageDataGenerator(dtype = 'float32', preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input)
val_datagen = ImageDataGenerator(dtype = 'float32', preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input)

img_size = 224

# train_datagen = ImageDataGenerator(dtype = 'float32', preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input)
# val_datagen = ImageDataGenerator(dtype = 'float32', preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input)

# img_size = 299

# Load datasets
train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    class_mode='categorical'
)

val_dataset = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    class_mode='categorical'
)

# Load ResNet50V2 with pretrained ImageNet weights, exclude the top layer
base_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_dataset.num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

print(model.summary())

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy','recall','precision'])

# Calculate steps per epoch
steps_per_epoch = train_dataset.samples // train_dataset.batch_size
validation_steps = val_dataset.samples // val_dataset.batch_size

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=10
)

model.save('Resnet50V2(TrafficSignNou)V2.h5')

# # Resnet 50 V2

# train_datagenV2 = ImageDataGenerator(dtype = 'float32', preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input)
# val_datagenV2 = ImageDataGenerator(dtype = 'float32', preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input)

# # Load datasets
# train_dataset = train_datagenV2.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),
#     class_mode='categorical'
# )

# val_dataset = val_datagenV2.flow_from_directory(
#     val_dir,
#     target_size=(224, 224),
#     class_mode='categorical'
# )

# # Load ResNet50V2 with pretrained ImageNet weights, exclude the top layer
# base_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# # Freeze the base model
# for layer in base_model.layers:
#     layer.trainable = False

# # Add custom layers for classification
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(train_dataset.num_classes, activation='softmax')(x)

# # Create the final model
# model = Model(inputs=base_model.input, outputs=predictions)

# print(model.summary())

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Calculate steps per epoch
# steps_per_epoch = train_dataset.samples // train_dataset.batch_size
# validation_steps = val_dataset.samples // val_dataset.batch_size

# # Train the model
# history = model.fit(
#     train_dataset,
#     validation_data=val_dataset,
#     steps_per_epoch=steps_per_epoch,
#     validation_steps=validation_steps,
#     epochs=10
# )

# model.save('ResNet50V2(TrafficSignNou)V2.keras')