#   -This script also uses the TrainingModel class but defines a more complex convolutional neural network architecture.-    #
#   -Architecture: Multiple convolutional layers with different filters and kernel sizes. Downsampling operations via strides in some convolutional layers. Additional convolutional layers that further extract features. Optional dense layers with dropout and L2 regularization after flattening the feature maps. A final dense output layer with softmax activation.-    #
#   -Uses the same general configuration as training_cp.py but with a richer network structure. This is the preferred option if you need a more robust model with better feature extraction and higher potential accuracy. It’s suited for more complex tasks and usually gives better performance.-    #
#   --    #
#   -Summary:Victor-Jnr-    #

import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from _training_model import TrainingModel
from tensorflow.keras.regularizers import l2
keras.regularizers.l2(0.01)

path = 'training/'

configs = {
    'dataset': {
        'training_images': "treatment/treated_data/training_images_dataset.npz",
        'training_labels': "treatment/treated_data/training_labels_dataset.npz",
        'testing_images': "treatment/treated_data/testing_images_dataset.npz",
        'testing_labels': "treatment/treated_data/testing_labels_dataset.npz"
    },
    'image': {
        'width': 28,
        'height': 28,
        'channels': 1
    },
    'nn_output': 30,
    'model': {
        'epochs': 200,
        'batch_size': 512
    },
    'path': {
        'chart': path + 'charts/',
        'model': path + 'model/',
        'history': path + 'history/'
    },
    'binary': False
}

nn = TrainingModel(configs)

training_arc = [
    {
        'conv2d1': {
            'filters': 32,
            'size': (3, 3)
        },
        'conv2d2': {
            'filters': 64,
            'size': (3, 3)
        },
        'conv2d3': {
            'filters': 32,
            'size': (3, 3)
        },
        'dense1': 768,
        'dense2': 192,
        'dense3': False,
        'lr': 0.001
    }

]

def classifier_func(training_arc):
    classifier = nn.instantiate_classifier()

    classifier.add(
        Conv2D(
            training_arc['conv2d1']['filters'],
            training_arc['conv2d1']['size'],
            padding='same',
            input_shape=(28, 28, 1), 
        )
    )

    classifier.add(Activation('relu'))

    #downsampling
    classifier.add(
        Conv2D(
            training_arc['conv2d1']['filters'],
            (3,3),
            padding='same',
            strides=(2,2)
        )
    )

    classifier.add(Activation('relu'))

    classifier.add(
        Conv2D(
            training_arc['conv2d2']['filters'],
            training_arc['conv2d2']['size'],
            padding='same'
        )
    )
    classifier.add(Activation('relu'))

    #downsampling
    classifier.add(
        Conv2D(
            training_arc['conv2d2']['filters'],
            (3,3),
            padding='same',
            activation='relu',
            strides=(2,2)
        )
    )

    classifier.add(
        Conv2D(
            training_arc['conv2d3']['filters'],
            training_arc['conv2d3']['size'],
            padding='same'
        )
    )

    classifier.add(Activation('relu'))

    classifier.add(Flatten())

    if training_arc['dense1']:
        classifier.add(Dense(units=training_arc['dense1'], activation='relu', activity_regularizer='l2'))
        classifier.add(Dropout(0.5))

    if training_arc['dense2']:
        classifier.add(Dense(units=training_arc['dense2'], activation='relu', activity_regularizer='l2'))
        classifier.add(Dropout(0.5))

    if training_arc['dense3']:
        classifier.add(Dense(units=training_arc['dense3'], activation='relu', activity_regularizer='l2'))
        classifier.add(Dropout(0.5))

    classifier.add(Dense(units = 30, activation = 'softmax')) 

    opt = optimizers.Adamax(learning_rate=training_arc['lr'], beta_1=0.9, beta_2=0.999)

    classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    classifier.summary()

    return classifier

nn.set_model(classifier_func, training_arc[0])
nn.train()
nn.save_model()
