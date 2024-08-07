from config import * 

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model

import numpy as np

def build_model(num_classes):
    """
    Builds a small image classifier using MobileNetV3Small backbone.

    Parameters:
      optimizer: AdamW
      loss: binary_crossentropy
      metrics: accuracy

    Args:
      num_classes: Number of classes for classification.

    Returns:
      A Keras model.
    """
    # Load pre-trained MobileNetV3Small model (without top layers)
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='gelu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
