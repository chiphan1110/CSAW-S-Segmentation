import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from config import *
from utils import flatten_
from batch_generator import batch_gen
from unet import Unet 
from resnetpp import ResUnetPlusPlus



def setup_environment(CUDA_VISIBLE_DEVICES):
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def prepare_datasets(data_path, n_patients, train_val_split):
    data_set = []
    for patient in os.listdir(data_path):
        curr = data_path + patient + "/"
        tmp = [curr + x for x in os.listdir(curr)]
        data_set.append(tmp)
    
    val1 = int(n_patients * train_val_split)
    train_set = data_set[:val1]
    val_set = data_set[val1:]

    train_set = flatten_(train_set)
    val_set = flatten_(val_set)
    return flatten_(train_set), flatten_(val_set)

def setup_model(input_shape, nb_classes, learning_rate, model_arch, spatial_dropout):
    if model_arch.lower() == "unet":
        network = Unet(input_shape=input_shape, nb_classes=nb_classes)
        network.encoder_spatial_dropout = spatial_dropout  
        network.decoder_spatial_dropout = spatial_dropout  
        network.set_convolutions([16, 32, 32, 64, 64, 128, 128, 256, 128, 128, 64, 64, 32, 32, 16])
        if IMG_SIZE == 1024:
            network.set_bottom_level(8)
        model = network.create()
    elif model_arch.lower() == "resunetpp":
        network = ResUnetPlusPlus(input_shape=input_shape, nb_classes=nb_classes)
        network.set_convolutions([16, 32, 64, 128, 256, 512])  # attempt to make it more shallow => perhaps won't overfit so easily? Perhaps I sould just use dropout
        model = network.create()
    else:
        raise ValueError("Unknown architecture selected. Please choose one of these: {'unet', 'resunetpp'}")

    model.compile(
        optimizer=Adam(learning_rate), 
        loss=network.get_dice_loss(use_background=USE_BACKGROUND)  
    )

    print(model.summary())  # prints the full architecture

    return model


def main():
    setup_environment(CUDA_VISIBLE_DEVICES)
    train_set, val_set = prepare_datasets(DATA_PATH, N_PATIENTS, TRAIN_VAL_SPLIT)
    n_train_steps = len(train_set) // BATCH_SIZE
    n_val_steps = len(val_set) // BATCH_SIZE
        
    train_gen = batch_gen(train_set, BATCH_SIZE, aug=TRAIN_AUG, class_names=CLASS_NAMES, input_shape=INPUT_SHAPE, epochs=N_EPOCHS, mask_flag=False, fine_tune=False)
    val_gen = batch_gen(val_set, BATCH_SIZE, aug=VAL_AUG, class_names=CLASS_NAMES, input_shape=INPUT_SHAPE, epochs=N_EPOCHS, mask_flag=False, fine_tune=False)
    
    checkpoint_cb = ModelCheckpoint(os.path.join(SAVE_PATH, "model_checkpoint.h5"), save_best_only=True)
    csv_logger_cb = CSVLogger(os.path.join(HISTORY_PATH, "training_log.csv"))
    model = setup_model(INPUT_SHAPE, NB_CLASSES, LEARNING_RATE, MODEL_ARCH, SPATIAL_DROPOUT)

    model.fit(
        train_gen,  
        steps_per_epoch=n_train_steps,
        epochs=N_EPOCHS,
        validation_data=val_gen,  
        validation_steps=n_val_steps,
        callbacks=[checkpoint_cb, csv_logger_cb]
    )

if __name__ == "__main__":
    main()