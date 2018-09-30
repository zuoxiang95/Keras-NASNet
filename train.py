# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from utils import get_random_data, get_classes
from keras.layers import Input, Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from nasnet import NASNetLarge


def acc(y_true, y_pred):
    index = tf.reduce_any(y_true > 0.5, axis=-1)
    res = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    index = tf.cast(index, tf.float32)
    res = tf.cast(res, tf.float32)
    return tf.reduce_sum(res * index) / (tf.reduce_sum(index) + 1e-7)


def train():
    data_file_path = 'shuf_train_file'
    log_dir = 'logs/'
    classes_path = 'my_class.txt'
    classes_name = get_classes(classes_path)
    num_classes = len(classes_name)
    batch_size = 32

    width = 331
    hight = 331

    base_model = NASNetLarge(input_shape=(width, hight, 3), weights='imagenet', include_top=False, pooling='avg')

    input_tensor = Input(shape=(None, None, 3))
    x = input_tensor
    # x = Lambda(preprocess_input)(x)
    x = base_model(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(input_tensor, x)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(data_file_path, 'r') as f1:
        lines = f1.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    if True:
        for i in range(len(model.layers) - 2):
            model.layers[i].trainable = False
        model.summary()
        model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['acc'])
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size=batch_size, input_shape=(width, hight),
                                                   num_classes=num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size=batch_size,
                                                                   input_shape=(width, hight), num_classes=num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=10,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save(log_dir + 'trained_weights_stage_1.h5')

    pass


def data_generator(data_lines, batch_size, input_shape, num_classes, random=True, verbose=False):
    """
        data generator for fit_generator.
    :param data_lines:
    :param batch_size:
    :param input_shape:
    :param num_classes:
    :return:
    """
    n = len(data_lines)
    i = 0
    while True:
        img_data = []
        label_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(data_lines)
            image, label = get_random_data(data_lines[i], input_shape, num_classes, random=random)
            img_data.append(image)
            label_data.append(label)
            i = (i + 1) % n
        img_data = np.asanyarray(img_data)
        label_data = np.asanyarray(label_data)
        yield img_data, label_data


def data_generator_wrapper(data_lines, batch_size, input_shape, num_classes, random=True, verbose=False):
    n = len(data_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(data_lines, batch_size, input_shape, num_classes, random, verbose)


if __name__ == '__main__':
    train()
