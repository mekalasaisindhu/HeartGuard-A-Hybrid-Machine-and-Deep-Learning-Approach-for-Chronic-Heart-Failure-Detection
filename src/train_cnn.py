import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from cnn_models import build_cnn
import os

def train_cnn_model(spectrogram_dir, model_out, epochs=10, batch_size=16):

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        spectrogram_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode="binary",
        subset="training"
    )

    val_data = datagen.flow_from_directory(
        spectrogram_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode="binary",
        subset="validation"
    )

    model = build_cnn()

    os.makedirs(os.path.dirname(model_out), exist_ok=True)

    checkpoint = ModelCheckpoint(
        model_out + ".keras",
        save_best_only=True,
        monitor='val_loss'
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[checkpoint, early_stop]
    )

    print("CNN model saved at:", model_out + ".keras")


if __name__ == "__main__":
    spectrogram_dir = "data/spectrogram"   # Folder with normal/abnormal
    model_out = "models/cnn/cnn_model"
    train_cnn_model(spectrogram_dir, model_out, epochs=10, batch_size=16)
