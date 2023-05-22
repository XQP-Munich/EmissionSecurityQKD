"""
This script handles the neural network.
"""
import datetime
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU support for TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np
import scipy as sp
import scipy.signal

import preprocessing as pp

# Set the seed for reproducibility
seed = 42
tf.random.set_seed(seed)
AUTOTUNE = tf.data.AUTOTUNE


def get_model_hyperparam_improved(technical_details, num_classes = 4):
    """
    Tensorflow 1d convolutional classifier model as obtained by hyperparameter optimization
    """
    samples_per_clock_cycle = technical_details["sample_freq"] // technical_details["clock_freq"]
    input_shape = samples_per_clock_cycle * (technical_details["steps_to_left"] + technical_details["steps_to_right"])
    
    filter1 = 13
    filter2 = 118
    filter3 = 100
    filtersize1 = 3
    filtersize2 = 15
    filtersize3 = 5
    poolsize1 = 2
    poolsize2 = 1
    poolsize3 = 4
    
    gaussian_noise = 0.1
    dropout = 0.25
    dense_size = 224
    
    activation = 'gelu'
    padding='causal'
    hp_learning_rate = 0.001
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((input_shape, 1)),
        
        layers.GaussianNoise(gaussian_noise),

        layers.Conv1D(filter1, filtersize1, activation=activation, padding=padding, dilation_rate=1),
        layers.MaxPooling1D(pool_size=poolsize1),
        layers.SpatialDropout1D(0.25),
        layers.BatchNormalization(),

        layers.Conv1D(filter2, filtersize2, activation=activation, padding=padding, dilation_rate=2),
        layers.MaxPooling1D(pool_size=poolsize2),
        layers.SpatialDropout1D(0.25),
        layers.BatchNormalization(),
        
        layers.Conv1D(filter3, filtersize3, activation=activation, padding=padding, dilation_rate=4),
        layers.MaxPooling1D(pool_size=poolsize3),
        layers.BatchNormalization(),

        layers.SpatialDropout1D(dropout),
        layers.Flatten(),
        layers.Dense(dense_size, activation=activation),

        layers.Dense(num_classes),
    ])

    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    return model


def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv1D(filter, 3, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv1D(filter, 3, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x


def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv1D(filter, 3, padding = 'same', strides = 2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv1D(filter, 3, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv1D(filter, 1, strides = 2)(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x


def create_model_sequential(
    technical_details, 
    num_classes=4,
    filter1=16, 
    filter2=64, 
    dense_size=512, 
    dropout=0.2, 
    gaussian_noise=0.15):
    """
    This method creates a Tensorflow model.
    """
    samples_per_clock_cycle = technical_details["sample_freq"] // technical_details["clock_freq"]
    input_shape = samples_per_clock_cycle * (technical_details["steps_to_left"] + technical_details["steps_to_right"])
    model_shape = (input_shape,)

    model = models.Sequential([
        layers.Input(shape=model_shape),
        layers.Reshape((*model_shape, 1)),

        layers.GaussianNoise(gaussian_noise),

        layers.Conv1D(filter1, 3, activation='relu'),
        layers.MaxPooling1D(),
        layers.BatchNormalization(),

        layers.Conv1D(filter2, 9, activation='relu'),
        layers.MaxPooling1D(),
        layers.BatchNormalization(),

        layers.Dropout(dropout),
        layers.Flatten(),
        layers.Dense(dense_size, activation='relu'),

        layers.Dense(num_classes),
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    return model


def create_model_residual(technical_details, dropout=0.2, num_classes=4, gaussian_noise=0.):
    """
    This method creates a Tensorflow model.
    """
    samples_per_clock_cycle = technical_details["sample_freq"] // technical_details["clock_freq"]
    input_shape = samples_per_clock_cycle * (technical_details["steps_to_left"] + technical_details["steps_to_right"])
    model_shape = (input_shape,)
    
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(model_shape)
    x = tf.keras.layers.GaussianNoise(gaussian_noise)(x_input)
    x = tf.keras.layers.Reshape((*model_shape, 1), input_shape=model_shape)(x)
    x = tf.keras.layers.ZeroPadding1D(3)(x)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling1D(2, padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(num_classes, activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x)
    
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )
    
    return model


def prepare_for_TensorFlow(data, labels, batch_size=32, validation_split=0.1, label_mapping=pp.translate_all):
    """
    This method makes the datasets compatible with TensorFlow.
    Additionally, a validation split is applied.
    """
    # Converts the label Strings into int.
    labels = label_mapping(labels)

    # Converts the numpy arrays into tensor slices
    complete_ds = tf.data.Dataset.from_tensor_slices((data, labels))
    complete_ds = complete_ds.shuffle(len(labels), reshuffle_each_iteration=False)

    # Makes a validation split
    num_test_samples = int(validation_split * len(labels))
    train_ds = complete_ds.skip(num_test_samples).shuffle(len(labels), reshuffle_each_iteration=True)
    val_ds = complete_ds.take(num_test_samples)

    # Batches the datasets
    val_ds = val_ds.batch(batch_size).cache().prefetch(AUTOTUNE)
    train_ds = train_ds.batch(batch_size).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds


class HaltThresholdCallback(tf.keras.callbacks.Callback):
    """Adapted from https://stackoverflow.com/a/59439972/9988487"""

    def __init__(self, metric, threshold, upper_bound=True):
        super().__init__()
        self.threshold = threshold
        self.metric = metric
        self.upper_bound = upper_bound

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        observed_value = logs.get(self.metric)
        if self.upper_bound ^ (observed_value <= self.threshold):
            print(f"\n\n\nCancelling training because reached {self.metric} {observed_value}. "
                  f"(threshold {self.threshold})\n\n\n")
            self.model.stop_training = True


def train_model(model, training_data, training_labels, machine_learning_settings,
                validation_data=None, validation_labels=None, batch_size=128,
                training_run_name=None,
                stop_at_val_acc=0.995, enable_tensorboard=False,early_stopping_patience=3,
                label_mapping=pp.translate_all,
                ):
    """
    This method trains the TensorFlow model.
    Unless specified, the validation data is obtained as a random split of training data.
    """
    if validation_data is None and validation_labels is None:
        # Makes the dataset compatible for training
        train_ds, val_ds = prepare_for_TensorFlow(training_data, training_labels, batch_size=batch_size,
                                                  validation_split=machine_learning_settings["validation_split"],
                                                 label_mapping=label_mapping)
    elif validation_data is not None and validation_labels is not None:
        train_ds = prepare_datasets_nosplit(training_data, training_labels, batch_size=batch_size,label_mapping=label_mapping)
        val_ds = prepare_datasets_nosplit(validation_data, validation_labels, batch_size=batch_size,label_mapping=label_mapping)
    else:
        raise ValueError("Give either both validation data and validation labels, or neither.")

    callbacks = [
        # stop training early when performance no longer improving
        tf.keras.callbacks.EarlyStopping(verbose=1, patience=early_stopping_patience),
        HaltThresholdCallback(metric='val_accuracy', threshold=stop_at_val_acc),
    ]

    if enable_tensorboard:
        # Tensorboard callback. See https://www.tensorflow.org/tensorboard/get_started
        # To view results, run `tensorboard --logdir ./paper/tensorboardlogs` (from `emmisionsecurity` folder)
        if training_run_name is None or os.path.exists(os.path.join("tensorboardlogs", training_run_name)):
            log_dir = os.path.join(
                "tensorboardlogs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), "UNNAMED_OR_EXISTING")
        else:
            log_dir = os.path.join("tensorboardlogs", training_run_name)
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

    # Trains the TensorFlow model
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=machine_learning_settings["training_EPOCHS"],
        callbacks=callbacks,
    )
    return history


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    return np.eye(nb_classes)[data.astype(int)]


def prepare_datasets_nosplit(data, labels, batch_size=32, label_mapping=pp.translate_all):
    """
    This method makes a dataset compatible with TensorFlow.
    """
    assert len(data) == len(labels), "Data/Labels length mismatch"

    # Convert the label Strings into int.
    labels = label_mapping(labels)

    # Convert the numpy arrays into tensor slices
    ds = tf.data.Dataset.from_tensor_slices((data, labels))

    # Specify that the dataset should be shuffled each training iteration
    ds = ds.shuffle(len(labels), reshuffle_each_iteration=True)

    # Batch the datasets
    ds = ds.batch(batch_size).cache().prefetch(AUTOTUNE)

    return ds


def test_only_most_certain_equal_proportions_of_predicted_labels(predicted_probabilities, true_labels, 
                           most_certain_fraction=0.01, selection_mode='max_probability'):
    
    classes = [0, 1, 2, 3]
    
    results = {}
    
    all_predictions = np.argmax(predicted_probabilities, axis=1)
    
    for i in classes:
        indices_predicted_i = (all_predictions == i).nonzero()
        
        if np.size(indices_predicted_i) == 0:
            fraction_i = 1.  # could also just abort here...
        else:
            # want to have equal numbers of predictions for each class.
            # however, the network is sometimes biased (why??) and predicts one class more often than others.
            # compensate for this behaviour and try to take an equal number of each predicted class.
            take_n_for_each_symbol = most_certain_fraction * np.size(all_predictions) / len(classes)
            fraction_i = take_n_for_each_symbol / np.size(indices_predicted_i)

        results[i] = test_only_most_certain(predicted_probabilities[indices_predicted_i],
                                            true_labels[indices_predicted_i],
                                            most_certain_fraction=fraction_i,
                                            selection_mode=selection_mode
                                            )
    print(f"Total test accuracy for 'most certain {100 * most_certain_fraction}% "
          f"of predictions when taking equal numbers of each predicted symbol: {np.mean(list(results.values()))}\n") 
    
    return results


def test_only_most_certain(predicted_probabilities, true_labels, 
                           most_certain_fraction=0.01, selection_mode='max_probability'):
    if selection_mode == 'entropy':
        # take the x% that have the least Shannon entropy.
        priorities = np.sum(
            -predicted_probabilities * np.log2(predicted_probabilities),
            axis=1)
    elif selection_mode == 'max_probability':
        # Take the x% that have the largest single probability
        priorities = -np.max(predicted_probabilities, axis=1)
    else:
        print(f"Invalid selection mode: {selection_mode}")
        return
    
    sorted_args = np.argsort(priorities)

    indices_to_evaluate = sorted_args[:int(most_certain_fraction * len(sorted_args))]

    test_labels = true_labels[indices_to_evaluate]
    predicted_labels = np.argmax(predicted_probabilities[indices_to_evaluate], axis=1)

    total_test_samples = len(test_labels)
    correct_test_samples = np.sum(predicted_labels == test_labels)

    # print("indices: ", indices_to_evaluate)
    # print(predicted_labels == test_labels)

    test_accuracy = correct_test_samples / total_test_samples
    print(f"Test accuracy for 'most certain {100 * most_certain_fraction}% "
      f"of predictions (total {len(predicted_labels)})': {test_accuracy}\n")

    confusion_matrix(test_labels, predicted_labels, comment=f"most certain {100 * most_certain_fraction}%")
    
    return test_accuracy


def test_model(model, test_data, test_labels, 
               test_start, technical_details, time_shift=0, label_mapping=pp.translate_all,
              most_certain_fraction=None,):
    """
    This method tests the TensorFlow model.
    """
    processed_test_data, processed_test_labels = pp.cut_data_in_pieces(test_data, test_labels, test_start, 1,
                       technical_details, data_augmentation_halflength=0, time_shift=time_shift,
    )

    predicted_probabilities = model.predict(processed_test_data)
    predicted_labels = np.argmax(predicted_probabilities, axis=1)
    
    # Translates the test labels from 'H','V','P','M' to 0,1,2,3
    test_labels = label_mapping(processed_test_labels)
    
    conf_mat = confusion_matrix(test_labels, predicted_labels)
    
    if label_mapping == pp.translate_all:
        for bit_mapping in ['0101', '0011', '0110']:
            print(f"Estimated bit prediction accuracy given mapping {bit_mapping}: {acc_bit_confmat(conf_mat, bit_mapping=bit_mapping)}")

    total_test_samples = len(test_labels)
    correct_test_samples = np.sum(predicted_labels == test_labels)

    test_accuracy = correct_test_samples / total_test_samples
    print(f"Test accuracy: {test_accuracy}\n")
    
    if most_certain_fraction:
        limited_test_accuracy = test_only_most_certain(
            predicted_probabilities, test_labels, most_certain_fraction=most_certain_fraction)
        
        most_certain_with_equal_label_proportions = test_only_most_certain_equal_proportions_of_predicted_labels(
            predicted_probabilities, test_labels, most_certain_fraction=most_certain_fraction)
        
        return {'test_accuracy': test_accuracy, 
                f'best_{most_certain_fraction}': limited_test_accuracy, 
                f'best_{most_certain_fraction}_eq_label_proportions': most_certain_with_equal_label_proportions}
    else:
        return test_accuracy


def test_model_all_displacements(model, test_data, test_labels, test_start, technical_details, time_shift=0, step_samples=1, label_mapping=pp.translate_all):
    """NOTE: Do not use! This is just for debugging time synchronization problems!"""
    samples_per_symbol = technical_details['sample_freq'] // technical_details['clock_freq']
    
    print(f"The method test_model_all_displacements is used. NOTE: The test accuracies might appear lower than the actually are. Only look at displacements that are higher than others but not at the absolute value!")
    best_test_acc = 0.0
    best_displacement = -1
    for displacement in range(-20, samples_per_symbol+20, step_samples):
        processed_test_data, processed_test_labels = pp.cut_data_in_pieces(test_data, test_labels, test_start, 1,
                       technical_details, data_augmentation_halflength=0, time_shift=time_shift+displacement,
        )
        
        # Lets the TensorFlow model predict the test data
        predicted_labels = np.argmax(model.predict(processed_test_data), axis=1)
        
        # Translates the test labels
        converted_test_labels = label_mapping(processed_test_labels)
        
        total_test_samples = len(converted_test_labels)
        correct_test_samples = np.sum(predicted_labels == converted_test_labels)

        test_accuracy = correct_test_samples / total_test_samples

        print(f" at displacement {displacement} samples, test_accuracy={test_accuracy}. \n", end='')
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_displacement = displacement
        
    print(f"\nBEST Test accuracy after trying all displacements: {best_test_acc:.4f} at displacement {best_displacement} while using a total of {len(predicted_labels)} datapoints")

    return best_test_acc


def confusion_matrix(true_labels, predicted_labels, comment=""):
    mat = tf.math.confusion_matrix(
    true_labels,
    predicted_labels,
    num_classes=4)
    print(f"Confusion matrix ({comment}):\nH V P M \n{mat}")
    
    return mat


def acc_bit_confmat(confusion_matrix, bit_mapping='0101'):
    """Estimate bit prediction probability given confusion matrix for symbols HVPM"""
    n_total_symbols = np.sum(confusion_matrix)
    n_correct_symbols = np.sum(np.diagonal(confusion_matrix))
    
    if bit_mapping == '0101':
        n_correct_bits = n_correct_symbols + confusion_matrix[0, 2] + confusion_matrix[2, 0] + confusion_matrix[3, 1] +confusion_matrix[1, 3]
    elif bit_mapping == '0011':
        n_correct_bits = n_correct_symbols + confusion_matrix[1, 0] + confusion_matrix[0, 1] + confusion_matrix[3, 2] +  confusion_matrix[2, 3]
    elif bit_mapping == '0110':
        n_correct_bits = n_correct_symbols + confusion_matrix[2, 1] + confusion_matrix[1, 2] + confusion_matrix[0, 3] +confusion_matrix[3, 0]
    else:
        raise ValueError(f"Invalid {bit_mapping=}. Expected string '0101', '0011' or '0110'")
    return n_correct_bits / n_total_symbols
