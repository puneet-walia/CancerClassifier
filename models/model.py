import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report

def load_data(file_path):
    name_class = os.listdir(file_path)
    filepaths = list(glob.glob(file_path + '/**/*.*'))
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepath = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    data = pd.concat([filepath, labels], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)
    
    return data

def balance_data(data):
    counts = data.Label.value_counts()
    sns.barplot(x=counts.index, y=counts)
    plt.xlabel('Type')
    plt.xticks(rotation=90)

    train, test = train_test_split(data, test_size=0.25, random_state=42)

    return train, test

def create_image_data_generators(train, test):
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train,
        x_col='Filepath',
        y_col='Label',
        target_size=(100, 100),
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=42
    )
    
    valid_gen = train_datagen.flow_from_dataframe(
        dataframe=test,
        x_col='Filepath',
        y_col='Label',
        target_size=(100, 100),
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
        seed=42
    )

    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test,
        x_col='Filepath',
        y_col='Label',
        target_size=(100, 100),
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    return train_gen, valid_gen, test_gen

def build_and_train_model(train_gen, valid_gen):
    pretrained_model = ResNet50(
        input_shape=(100, 100, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    pretrained_model.trainable = False

    inputs = pretrained_model.input
    x = Dense(128, activation='relu')(pretrained_model.output)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    my_callbacks = [EarlyStopping(monitor='val_accuracy', min_delta=0, patience=2, mode='auto')]

    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=100
    )

    return model, history

def evaluate_model(model, test_gen):
    results = model.evaluate(test_gen, verbose=0)

    print("    Test Loss: {:.5f}".format(results[0]))
    print("Test Accuracy: {:.2f}%".format(results[1] * 100))

    # Predict the label of the test_gen
    pred = model.predict(test_gen)
    pred = np.argmax(pred, axis=1)

    # Map the label
    labels = (train_gen.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    pred = [labels[k] for k in pred]

    y_test = list(test.Label)
    print(classification_report(y_test, pred))

def main():
    file_path = '/content/drive/MyDrive/scdd/scdd/archive (2)/data/train'
    data = load_data(file_path)
    train, test = balance_data(data)
    train_gen, valid_gen, test_gen = create_image_data_generators(train, test)
    model, history = build_and_train_model(train_gen, valid_gen)
    evaluate_model(model, test_gen)

if __name__ == "__main__":
    main()
