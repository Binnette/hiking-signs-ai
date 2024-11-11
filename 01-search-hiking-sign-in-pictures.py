import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory

# Define constants
SEED = 123
EPOCHS = 10
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32

# Set up logging
logging.basicConfig(filename='errors.log', level=logging.ERROR)

# Define paths
training_dir = '/mnt/Work/wk/ia-hiking-signs/photos'
testing_dir1 = '/mnt/Work/Images/Camera' # '/home/binnette/Images/'
results = 'images_new.txt'

# Ensure TensorFlow uses CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
physical_devices = tf.config.experimental.list_physical_devices('CPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

# Remove results file if it exists
if os.path.exists(results):
    os.remove(results)

def trainModel():
    # Load and preprocess the training data
    train_dataset = image_dataset_from_directory(
        training_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    validation_dataset = image_dataset_from_directory(
        training_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # Build the model
    model = models.Sequential([
        layers.Rescaling(1./255),
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(len(train_dataset.class_names), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )

    return model, train_dataset

# Function to predict and display results
def predict_and_display(model, train_dataset, image_dir, results, target_class=None, min_confidence=0):
    with open(results, 'a') as txt:
        for subdir, dirs, files in os.walk(image_dir):
            for file in files:
                try:
                    image_path = os.path.join(subdir, file)
                    img = keras.preprocessing.image.load_img(
                        image_path, target_size=IMAGE_SIZE
                    )
                    img_array = keras.preprocessing.image.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

                    predictions = model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])
                    confidence = 100 * np.max(score)
                    class_name = train_dataset.class_names[np.argmax(score)]

                    # Print and write to file if the predicted class matches the target class and confidence is above the minimum
                    if (target_class is None or class_name == target_class) and confidence > min_confidence:
                        print(f"Path: {image_path} Class: {class_name} Confidence: {confidence:.2f}%")
                        txt.write(image_path + '\n')
                        txt.flush()
                except Exception as e:
                    logging.error(f"Error on file: {image_path} => {e}")

def main():
    # Train the model
    model, train_dataset = trainModel()

    # Run predictions on new images
    predict_and_display(model, train_dataset, testing_dir1, results, target_class="HikingSigns", min_confidence=70)

if __name__ == "__main__":
    main()
