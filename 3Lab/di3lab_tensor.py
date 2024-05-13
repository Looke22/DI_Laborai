import numpy as np
import keras
import tensorflow as tf
from keras import layers
from keras.api.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

"""
## Prepare the data
"""
#region Prepare the data
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
#endregion
"""
## Data augmentationp
"""
#region Data augmentation
data_augmentation_layers = [
    #layers.RandomFlip("horizontal"),
    #layers.RandomFlip("vertical"),
    layers.RandomRotation(factor=0.05),
    layers.RandomContrast(factor=0.005),
    layers.RandomZoom(width_factor=0.2, height_factor=0.2, fill_mode="reflect")
    #layers.RandomBrightness(factor=0.01),
    #layers.ColorJitter(saturation=0.2, hue=0.2),
    #layers.RandomCrop(height=28, width=28)
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

x_train_augmented = data_augmentation(x_train)

#endregion

"""
Visualization
"""
#region Visualization
# Plot original and augmented images
def plot_augmented_images(original_images, augmented_images, labels, num_images=3):
    plt.figure(figsize=(15, 6))
    
    # Select random indices
    num_total_images = original_images.shape[0]
    random_indices = np.random.choice(num_total_images, size=num_images, replace=False)
    
    for i, idx in enumerate(random_indices):
        # Original image
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[idx].reshape(28, 28), cmap='gray_r')
        plt.title("Original: {}".format(int(labels[idx])))
        plt.axis("off")
        
        # Augmented image
        ax = plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(augmented_images[idx].numpy().reshape(28, 28), cmap='gray_r')
        plt.title("Augmented")
        plt.axis("off")
    plt.show()
    
plot_augmented_images(x_train, x_train_augmented, y_train)
#endregion



"""
## Build the model
"""
#region Build the model
x_train_combined = np.concatenate((x_train, x_train_augmented), axis=0)

y_train = keras.utils.to_categorical(y_train, num_classes)
# Concatenate and one-hot encode the combined target labels
y_train_combined = np.concatenate((y_train, y_train), axis=0)

# convert class vectors to binary class matrices
y_test = keras.utils.to_categorical(y_test, num_classes)


model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(2, 2), activation="relu", padding="same"),
        layers.Conv2D(64, kernel_size=(2, 2), activation="relu", padding="valid"),
        layers.Dropout(0.5),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="valid"),
        layers.Conv2D(64, kernel_size=(2, 2), activation="relu", padding="valid"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="valid"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="valid"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ]
)

model.summary()
#endregion
"""
## Train the model
"""
batch_size = 128
epochs = 20

# Define the file paths for saving the models
best_model_checkpoint_path = 'best_model.keras'
last_model_checkpoint_path = 'last_model.keras'

# Define a callback to save the model with the best validation accuracy
best_model_checkpoint = ModelCheckpoint(filepath=best_model_checkpoint_path,
                                       monitor='val_loss',
                                       save_best_only=True,
                                       mode='min',
                                       verbose=1)

# Define a callback to save the last model after training
last_model_checkpoint = ModelCheckpoint(filepath=last_model_checkpoint_path,
                                        save_weights_only=False,
                                        verbose=1)


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

"""

"""
model.fit(x_train_combined, y_train_combined, 
          batch_size=batch_size, 
          epochs=epochs, validation_split=0.1, 
          callbacks=[best_model_checkpoint, last_model_checkpoint])




"""
## Evaluate the trained model
"""
best_model = keras.models.load_model(best_model_checkpoint_path)
last_model = keras.models.load_model(last_model_checkpoint_path)

score = model.evaluate(x_test, y_test, verbose=0)
best_score = best_model.evaluate(x_test, y_test, verbose=0)

print("LAST MODEL:")
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print(" ")
print("BEST MODEL:")
print("Test loss:", best_score[0])
print("Test accuracy:", best_score[1])

"""
Plot predictions
"""

def plot_predictions(model, images, num_images=3):
    plt.figure(figsize=(15, 9))
    num_total_images = images.shape[0]

    # Predictions for all images
    predictions = model.predict(images)

    # Get indices of best predictions
    best_indices = np.argsort(np.max(predictions, axis=1))[-num_images:][::-1]

    # Get indices of worst predictions
    worst_indices = np.argsort(np.max(predictions, axis=1))[:num_images]

    # Plot best predictions in the first row
    for i, idx in enumerate(best_indices):
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(images[idx].reshape(28, 28), cmap='gray_r')

        # Get top 3 classes and their probabilities
        top_classes = np.argsort(-predictions[idx])[:3]
        top_probs = predictions[idx][top_classes]

        # Format title text
        title_text = '\n'.join([f'Class: {cls}, Probability: {prob:.2%}' for cls, prob in zip(top_classes, top_probs)])

        plt.title(title_text, color='#017653')
        plt.axis("off")

    # Plot worst predictions in the second row
    for i, idx in enumerate(worst_indices):
        ax = plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(images[idx].reshape(28, 28), cmap='gray_r')

        # Get top 3 classes and their probabilities
        top_classes = np.argsort(-predictions[idx])[:3]
        top_probs = predictions[idx][top_classes]

        # Format title text
        title_text = '\n'.join([f'Class: {cls}, Probability: {prob:.2%}' for cls, prob in zip(top_classes, top_probs)])

        plt.title(title_text, color='red')  # Highlight worst predictions in red
        plt.axis("off")

    plt.show()

# Display the prediction results including probability ratings
plot_predictions(best_model, x_test, y_test)
