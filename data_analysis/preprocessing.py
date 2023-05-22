"""
This script preprocesses the data for the machine learning script and displays the performance of the neural network.
"""
import os
import re
import csv
import json
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import scipy as sp
import scipy.signal
from scipy import fft

from verify_dataset_hash import get_repo_sha1_and_diff


# Set the seed for reproducibility
seed = 42
np.random.seed(seed)
plt.rcParams['figure.figsize'] = [15, 10]


def load_ocilloscope_csv_signal(csv_path: str):
    """Loads csv file as output by Lecroy oscilloscope and returns signal. Ignores time information (first column)."""
    with open(csv_path, 'r') as f:
        assert f.readline()[:6] == "LECROY", f"Given file {csv_path} is not a csv file measured by Lecroy Oscilloscope."
        osci_time_signal = np.genfromtxt(f,
                                         delimiter=',', skip_header=4)  # Note: readline() skips the first header line
    return osci_time_signal[:, 1]  # take only signal (dimension 1), ignore time (dimension 0).


def load_labels_from_keyfile(file_path):
    labels = []
    with open(file_path, 'r') as f:
        assert f.readline()[:8] == "#Keyfile", f"Given file {file_path} is not a valid keyfile."
        opened = f.readlines()[2:]
        for element in opened:
            labels.append(element.strip())
    return labels


def has_npy_files(folder_path) -> bool:
    all_files = os.listdir(folder_path)
    has_npy = False
    for f in all_files:
        if f.endswith(".npy") or f.endswith(".npz"):
            has_npy = True
            break
    return has_npy


def csv_path_to_npy_path(file_path: str) -> str:
    folder_path = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    if file_name.endswith("_labels.txt"):
        return os.path.join(folder_path, file_name[:-4] + "_numpy.npy")
    elif file_name.endswith(".txt") or file_name.endswith(".csv"):
        return os.path.join(folder_path, file_name[:-4] + "_numpy.npy")
    else:
        warnings.warn(f"Conversion to numpy (for caching) not specified for path {file_name}.")


def load_cached_data_file(file_path, verbose=False):
    assert os.path.exists(file_path), f"File {file_path} does not exist."

    file_name = os.path.basename(file_path)
    if file_path.endswith('.npy'):
        return np.load(file_path)
    else:
        cached_file_path = csv_path_to_npy_path(file_path)
        if os.path.exists(cached_file_path):
            return np.load(cached_file_path)
        elif file_name.endswith("_labels.txt"):
            labels = load_labels_from_keyfile(file_path)
            np.save(cached_file_path, labels)
            if verbose:
                print(f"Converted labels file {file_path} to Numpy .npy at {cached_file_path}.")
            return labels
        elif file_name.endswith(".txt") or file_name.endswith(".csv"):
            signal = load_ocilloscope_csv_signal(file_path)
            np.save(cached_file_path, signal)
            if verbose:
                print(f"Converted time trace file {file_path} to Numpy .npy at {cached_file_path}.")
            return signal
        else:
            if verbose:
                print(f"No loading or conversion specified for {file_path}.")


def convert_osci_files_to_npy(folder_path, verbose=False) -> None:
    warnings.warn("This function is deprecated. Use `load_cached_data_file` on individual files!")
    for name in os.listdir(folder_path):
        try:
            if name.endswith("_labels.txt"):
                labels = load_labels_from_keyfile(os.path.join(folder_path, name))
                np.save(os.path.join(folder_path, name[:-4] + "_numpy.npy"), labels)
                if verbose:
                    print(f"Converted labels file {name} to Numpy .npy format.")
            elif name.endswith(".txt") or name.endswith(".csv"):
                signal = load_ocilloscope_csv_signal(os.path.join(folder_path, name))
                np.save(os.path.join(folder_path, name[:-4] + "_numpy.npy"), signal)
                if verbose:
                    print(f"Converted time trace file {name} to Numpy .npy format.")
            else:
                if verbose:
                    print(f"NOT CONVERTING file {name}.")
        except Exception as e:
            print(f"Failed to convert csv file {name} and save to Numpy file. Error: {e}")


def translation(letter, forward=True):
    """
    This method translates:
    H, V, P, M
    0, 1, 2, 3
    This is necessary because TensorFlow does not accept Strings as classes.
    """
    if forward:
        if letter == "H":
            return 0
        elif letter == "V":
            return 1
        elif letter == "P":
            return 2
        elif letter == "M":
            return 3
        else:
            return letter
    else:
        if letter == 0:
            return "H"
        elif letter == 1:
            return "V"
        elif letter == 2:
            return "P"
        elif letter == 3:
            return "M"
        else:
            return letter


def translate_all(labels, forward=True):
    """
    This method converts an array of Strings to a format that is acceptable for TensorFlow.
    """
    translated = np.zeros(len(labels), dtype=int)
    for i in range(len(labels)):
        translated[i] = translation(labels[i], forward=forward)
    return translated


def translate_all_onlybit(labels, bit_mapping='0101'):
    """
    Converts a 1d array of string labels (chosen from 'H', 'V', 'P', 'M') to bit values as specified by bit_mapping.

    Only possible to specify 3 of the 6 possible assignments, where two 0s and two 1s are used.
    This is because swapping zeros and ones is considered a trivial operation
    """
    num_labels = translate_all(labels, forward=True)
    if bit_mapping == '0101':
        bits = (num_labels % 2).astype(int)
    elif bit_mapping == '0011':
        bits = np.floor(num_labels / 2).astype(int)
    elif bit_mapping == '0110':
        bits = np.logical_or(num_labels == 1, num_labels == 2).astype(int)
    else:
        raise ValueError("Invalid {bit_mapping=}. Expected one of '0101', '0011', '0110'.")

    return bits


def find_repeat_debug_symbol_position(repeat_debug_signal, expected_pulse_width, offset_header):
    # find header position
    # Find the biggest overlap between header in the laser driver and the header model
    correlation = sp.signal.correlate(
        np.abs(np.median(repeat_debug_signal) - repeat_debug_signal),
        np.ones(expected_pulse_width), mode="valid")
    symbol_sequence_starts_at_sample = np.argmax(correlation) + offset_header

    # try to decide if finding the header worked
    if np.max(correlation) / np.median(correlation) < 2.:
        warnings.warn("Finding the header likely failed!")

    return symbol_sequence_starts_at_sample


def debug_plot_for_synchronization(training_data, training_synch_signal,
                                   repeat_sig_peak_width_samples, symbol_sequence_starts_at_sample=0):
    """Not tested"""
    fig, ax = plt.subplots()
    ax.plot(training_data, label="signal")
    # ax.plot(training_synch_signal / np.max(training_synch_signal))
    # ax.plot(training_clock_signal / np.max(training_clock_signal))
    ax.axvline(x=symbol_sequence_starts_at_sample, color='red')
    ax.legend()
    plt.show()
    # #########################################
    # rectangle = np.hstack(
    #    (np.zeros(symbol_sequence_starts_at_sample), np.ones(repeat_sig_peak_width_samples),
    #     np.zeros(len(training_data) - symbol_sequence_starts_at_sample - repeat_sig_peak_width_samples)))
    # plot_header_laser_driver(training_synch_signal, rectangle)


def load_data_synchronized(folder_path,
                           file_path,
                           file_path_synch,
                           clock_freq,  # Clock frequency the sender unit operates on
                           sample_freq,  # Sample frequency the oscilloscope uses
                           signal_length,  # Total number of measurement points
                           offset_header,
                           do_normalize_data=False,
                           debug=False,
                           prefix="training",
                           **kwargs,  # are not used, only warning is emitted
                           ):
    """
    Loads data from given folder. Uses the debug repetition signal to synchronize training data with training labels.
    Does NOT synchronize test data and labels.
    """
    kwargs = {}  # Make sure they are not used

    probe_signal = load_cached_data_file(file_path, verbose=debug)
    labels = load_cached_data_file(os.path.join(folder_path, prefix + "_labels.txt"), verbose=debug)

    # training_clock_signal = np.load(os.path.join(folder_path, prefix + "_clock_debug_signal_numpy.npy"))
    synch_signal = load_cached_data_file(file_path_synch, verbose=debug)

    expected_repeat_pulse_width = sample_freq // clock_freq
    symbols_start_correlation = find_repeat_debug_symbol_position(synch_signal, expected_repeat_pulse_width, offset_header)

    # Adjust symbols_start_at_sample to the closest zero crossing from negative to positive in the clock
    clock = find_clock(probe_signal, clock_freq, sample_freq, signal_length)
    zero_crossings = np.where(np.diff(np.signbit(clock)))[0]
    symbols_start_at_sample = zero_crossings[np.argmin(np.abs(zero_crossings - symbols_start_correlation))] + 1
    if np.signbit(clock[symbols_start_at_sample + 1]):
        symbols_start_at_sample = zero_crossings[np.argmin(np.abs(zero_crossings - symbols_start_correlation)) + 1] + 1

    # Note: this "starting point" is NOT synchronized to the device clock. Three ways to do it:
    # 1. Do synchronize it to the device clock by taking the clock from the probe signal
    # 2. Add a constant offset that is known empirically
    #       (assumes that difference between repeat signal and clock phase is constant)
    # 3. Do not synchronize to the clock at all.
    #       Use repeat signal in test measurement as well, or try all displacements.

    print(f"Successfully loaded data from folder {os.path.abspath(folder_path)}")

    if debug:
        debug_plot_for_synchronization(probe_signal, synch_signal,
                                       expected_repeat_pulse_width, symbols_start_at_sample)

    if do_normalize_data:
        probe_signal = normalized_1d_array(probe_signal)

    return probe_signal, labels, symbols_start_at_sample,


def load_test_data_synchronized(folder_path,
                                clock_freq,  # Clock frequency the sender unit operates on
                                sample_freq,  # Sample frequency the oscilloscope uses
                                signal_length,  # Total number of measurement points
                                force_create_npy_files=False,
                                do_normalize_data=False,
                                data_index=0,
                                offset=-100,
                                ):
    """
    Loads synchronized test data from given folder.
    """
    if force_create_npy_files or not has_npy_files(folder_path):
        convert_osci_files_to_npy(folder_path)

    filename = [filename for filename in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, filename))]

    pattern_data = re.compile("test_(.*)_data_numpy.npy")
    filename_data = [pattern_data.match(i)[0] for i in filename if pattern_data.match(i) != None]
    pattern_synch = re.compile("test_(.*)_synch_debug_signal_numpy.npy")
    filename_synch = [pattern_synch.match(i)[0] for i in filename if pattern_synch.match(i) != None]

    filename_data.sort(key=natural_keys)
    filename_synch.sort(key=natural_keys)

    test_data = np.load(os.path.join(folder_path, filename_data[data_index]))
    test_labels = np.load(os.path.join(folder_path, "test_labels_numpy.npy"))
    synch_signal = np.load(os.path.join(folder_path, filename_synch[data_index]))

    print(f"Successfully loaded test data from folder {os.path.abspath(folder_path)}")

    expected_repeat_pulse_width = sample_freq // clock_freq
    symbols_start_correlation = find_repeat_debug_symbol_position(synch_signal, expected_repeat_pulse_width)

    # Adjust symbols_start_at_sample to the closest zero crossing from negative to positive in the clock
    clock = find_clock(test_data, clock_freq, sample_freq, signal_length)
    zero_crossings = np.where(np.diff(np.signbit(clock)))[0]
    symbols_start_at_sample = zero_crossings[np.argmin(np.abs(zero_crossings - symbols_start_correlation))] + 1
    if np.signbit(clock[symbols_start_at_sample + 1]):
        symbols_start_at_sample = zero_crossings[np.argmin(np.abs(zero_crossings - symbols_start_correlation)) + 1] + 1

    symbols_start_at_sample = symbols_start_at_sample
    test_data = test_data[symbols_start_at_sample + offset:]

    if do_normalize_data:
        test_data = normalized_1d_array(test_data)

    return test_data, test_labels


def normalized_1d_array(arr):
    meanval_train = np.mean(arr)
    stdval_train = np.std(arr)

    return (arr - meanval_train) / stdval_train


def find_clock(signal, clock_freq, sample_freq, signal_length):
    """
    This method returns the clock component from the signal using custom Fourier filtering.
    """
    fft = sp.fft.fft(signal)
    pos_freq_clock = clock_freq * signal_length // sample_freq
    filter = np.zeros(fft.shape)
    filter[pos_freq_clock] = 1
    filter[-pos_freq_clock] = 1
    clock = np.real(sp.fft.ifft(filter * fft))

    return clock


# def find_header(training_data, laser_driver, technical_details, debug=False):
#     """
#     This method finds the header in the laser driver signal.
#     The position for the header is corrected to the next zero crossing of the clock frequency found in the training data.
#     """
#     # Normalizing the signals
#     training_data_copy = training_data / np.max(training_data)
#     laser_driver_copy = laser_driver / np.max(laser_driver)

#     # Sets the laser driver signal to zero in every position except the peaks
#     # The trigger level is the threshold that identifies a peak
#     trigger_level = 0.6
#     laser_driver_copy[laser_driver_copy < trigger_level] = 0
#     laser_driver_copy[laser_driver_copy >= trigger_level] = 1

#     # Creates the header model
#     header_length = technical_details["header_length"] * technical_details["sample_freq"] // technical_details[
#         "clock_freq"]
#     rectangle = np.ones(header_length)

#     # Find the biggest overlap between header in the laser driver and the header model
#     correlation = sp.signal.correlate(laser_driver_copy, rectangle, mode="valid")
#     header_position_correlation = np.argmax(correlation)

#     # try to decide if finding the header worked
#     if np.max(correlation) / np.median(correlation) < 2.:
#         warnings.warn("Finding the header likely failed!")

#     # Adjust the header position to the closest zero crossing from negative to positive in the clock
#     clock = find_clock(training_data_copy, technical_details)
#     zero_crossings = np.where(np.diff(np.signbit(clock)))[0]
#     header_position = zero_crossings[np.argmin(np.abs(zero_crossings - header_position_correlation))] + 1
#     if np.signbit(clock[header_position + 1]):
#         header_position = zero_crossings[np.argmin(np.abs(zero_crossings - header_position_correlation)) + 1] + 1

#     print(f"Found the header position at: {header_position}")

#     if debug:
#         rectangle = np.hstack(
#             (np.zeros(header_position), np.ones(header_length),
#              np.zeros(len(training_data_copy) - header_position - header_length)))
#         plot_header_signal(training_data_copy, clock, rectangle)
#         plot_header_laser_driver(laser_driver_copy, rectangle)

#     return header_position


def get_datasets(
        folder_path,
        technical_details,
        offset_header,
        data_augmentation_halflength=0,
        force_create_npy_files=False,
        debug=False,
        time_shift=0,
        max_train_files=None,
        ):

    if force_create_npy_files or not has_npy_files(folder_path):
        convert_osci_files_to_npy(folder_path)

    filenames = [filename for filename in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, filename))]

    pattern_data = re.compile("train_(.*)_data_numpy.npy")
    filename_data = [pattern_data.match(i)[0] for i in filenames if pattern_data.match(i) is not None]
    pattern_synch = re.compile("train_(.*)_synch_debug_signal_numpy.npy")
    filename_synch = [pattern_synch.match(i)[0] for i in filenames if pattern_synch.match(i) is not None]

    filename_data.sort(key=natural_keys)
    filename_synch.sort(key=natural_keys)

    filename_data = filename_data[:max_train_files]
    filename_synch = filename_synch[:max_train_files]

    validation_data, validation_labels, validation_start = load_data_synchronized(
        folder_path=folder_path,
        file_path=os.path.join(folder_path, filename_data[0], ),
        file_path_synch=os.path.join(folder_path, filename_synch[0], ),
        clock_freq=technical_details["clock_freq"],
        sample_freq=technical_details["sample_freq"],
        signal_length=technical_details["signal_length"],
        offset_header=offset_header,
        do_normalize_data=technical_details["do_normalize_data"],
        prefix="training",
        debug=debug,
    )

    validation_data, validation_labels = cut_data_in_pieces(
        validation_data, validation_labels, validation_start, technical_details["header_length"], technical_details, time_shift=time_shift,
        data_augmentation_halflength=0,
    )
    len_set = validation_data.shape[0]
    all_data = np.zeros((len(filename_data) * len_set * (data_augmentation_halflength + 1) * 2, 500))
    all_labels = np.zeros((len(filename_data) * len_set * (data_augmentation_halflength + 1) * 2,), dtype=str)

    index = 0

    for dataname, synchname in zip(filename_data[1:], filename_synch[1:]):

        training_data, training_labels, train_start = load_data_synchronized(
            folder_path=folder_path,
            file_path=os.path.join(folder_path, dataname, ),
            file_path_synch=os.path.join(folder_path, synchname, ),
            clock_freq=technical_details["clock_freq"],
            sample_freq=technical_details["sample_freq"],
            signal_length=technical_details["signal_length"],
            offset_header=offset_header,
            do_normalize_data=technical_details["do_normalize_data"],
            prefix="training",
            debug=debug,
        )

        processed_training_data, processed_training_labels = cut_data_in_pieces(
            training_data, training_labels, train_start, technical_details["header_length"], technical_details, time_shift=time_shift,
            data_augmentation_halflength=data_augmentation_halflength,
        )

        increment_index = processed_training_data.shape[0]

        all_data[index:index + increment_index, :] = processed_training_data
        all_labels[index:index + increment_index] = processed_training_labels
        index += increment_index

    processed_training_data = all_data[:index]
    processed_training_labels = all_labels[:index]

    return processed_training_data, processed_training_labels, validation_data, validation_labels,


def load_test_datasets(
        folder_path,
        technical_details,
        offset_header,
        debug=False,
        data_index=0,
        ):
    filename = [filename for filename in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, filename))]

    pattern_data = re.compile("test_(.*)_data_numpy.npy")
    filename_data = [pattern_data.match(i)[0] for i in filename if pattern_data.match(i) != None]
    pattern_synch = re.compile("test_(.*)_synch_debug_signal_numpy.npy")
    filename_synch = [pattern_synch.match(i)[0] for i in filename if pattern_synch.match(i) != None]

    filename_data.sort(key=natural_keys)
    filename_synch.sort(key=natural_keys)

    test_data, test_labels, test_start = load_data_synchronized(
        folder_path=folder_path,
        file_path=os.path.join(folder_path, filename_data[data_index], ),
        file_path_synch=os.path.join(folder_path, filename_synch[data_index], ),
        clock_freq=technical_details["clock_freq"],
        sample_freq=technical_details["sample_freq"],
        signal_length=technical_details["signal_length"],
        offset_header=offset_header,
        do_normalize_data=technical_details["do_normalize_data"],
        prefix="test",
        debug=debug,
    )

    return test_data, test_labels, test_start


def cut_data_in_pieces(training_data, training_labels, header_position, header_length,
                       technical_details, time_shift=0, data_augmentation_halflength=0,):
    """
    This method cuts the training data into 5 symbol pieces starting at the header.
    It selects a window of 5 symbols and shifts the next window by 1 symbol.
    """
    step_size = technical_details["sample_freq"] // technical_details["clock_freq"]
    num_labels = len(training_labels)
    steps_to_left = technical_details["steps_to_left"]
    steps_to_right = technical_details["steps_to_right"]

    # This applies a time shift on the header position (not relevant for main_distance)
    header_position += time_shift

    # Calculates the end position of the header
    header_end_position = header_position + header_length * step_size

    # The data is separated in a part before and a part after the header
    if header_position < num_labels * step_size:
        data_before_header = training_data[:header_position]
    else:
        data_before_header = training_data[header_position - num_labels * step_size:header_position]

    if len(training_data) - header_end_position < num_labels * step_size:
        data_after_header = training_data[header_end_position:]
    else:
        data_after_header = training_data[header_end_position:header_end_position + num_labels * step_size]

    # Creates arrays for the 5 symbol pieces and their label
    res_signal_before = []
    res_labels_before = ""
    res_signal_after = []
    res_labels_after = ""

    # Starts with the 4th symbol after the header to avoid running into index issues
    # First, this checks if the cut will be fully contained in the data left
    # If successful this cuts a 5 symbol long piece out of the training data
    # Data augmentation is introduced via a random shift in the cut that is added additionally
    # The cut is labeled with the corresponding polarization and the entire process is repeated with a shift of 1 symbol
    for i in range(steps_to_left + 1, num_labels - steps_to_right):
        for augmentation in range(-data_augmentation_halflength, data_augmentation_halflength + 1, 1):
            if (i + steps_to_right) * step_size + augmentation <= len(data_after_header):
                seq_after_header = data_after_header[
                                   (i - steps_to_left) * step_size + augmentation
                                   :
                                   (i + steps_to_right) * step_size + augmentation]
                label = training_labels[i]
                res_signal_after.append(seq_after_header)
                res_labels_after += label
            else:
                break

    for i in range(steps_to_right + 1, num_labels - steps_to_left):
        for augmentation in range(-data_augmentation_halflength, data_augmentation_halflength + 1, 1):
            if (i + steps_to_right) * step_size + augmentation <= len(data_before_header):
                seq_before_header = data_before_header[
                                    -((i + 1 + steps_to_left) * step_size + augmentation)
                                    :
                                    -((i + 1 - steps_to_right) * step_size + augmentation)]
                label = training_labels[-(i + 1)]
                res_signal_before.append(seq_before_header)
                res_labels_before += label
            else:
                break

    print(f"Cut data in pieces with these shapes: "
          f"after header:{np.array(res_signal_after).shape}, before header:{np.array(res_signal_before).shape}")

    res_signal_after = np.array(res_signal_after)
    res_signal_before = np.array(res_signal_before)

    # checks if a vstack can be performed
    if len(res_signal_after.shape) == 1:
        complete_signal = res_signal_before
    elif len(res_signal_before.shape) == 1:
        complete_signal = res_signal_after
    else:
        complete_signal = np.vstack((res_signal_after, res_signal_before))
    complete_label = res_labels_after + res_labels_before
    complete_label = np.array(list(complete_label))

    return complete_signal, complete_label


def cut_test_in_pieces(test_data, technical_details, time_shift=0, synch_to_clock=False):
    """
    This method cuts the test data into 5 symbol pieces starting at the header.
    It selects a window of 5 symbols and shifts the next window by 1 symbol.
    """
    step_size = technical_details["sample_freq"] // technical_details["clock_freq"]
    count_test_data = technical_details["signal_length"] // step_size
    steps_to_left = technical_details["steps_to_left"]
    steps_to_right = technical_details["steps_to_right"]

    if synch_to_clock:
        # Filter the clock component of the signal
        clock = find_clock(test_data, technical_details)

        # The first zero crossing from negative to positive marks the starting position for the cut.
        zero_crossings = np.where(np.diff(np.signbit(clock)))[0]
        start_cut = zero_crossings[0]
        if np.signbit(clock[start_cut + 1]):
            start_cut = zero_crossings[1]
    else:
        start_cut = 0
    # This applies a time shift on the starting position for the cut (not relevant for main_distance)
    start_cut += time_shift

    # Makes sure the index is positive by shifting the starting position
    while start_cut < 0:
        start_cut += step_size

    # Discards the part of the data that is not within a full period of the clock.
    test_data = test_data[start_cut:-(100 - start_cut % 100)]

    samples_in_a_piece = (steps_to_left + steps_to_right) * step_size
    num_pieces = int(np.floor(
        1 + (len(test_data) - samples_in_a_piece) / step_size
    ))
    # Create array for the pieces of `(steps_to_left + steps_to_right)` (usually 5) symbols
    complete_pattern = np.zeros((num_pieces, samples_in_a_piece))
    total_steps = steps_to_left + steps_to_right
    for i in range(count_test_data):
        if (i + total_steps) * step_size <= len(test_data):
            complete_pattern[i] = test_data[i * step_size:(i + total_steps) * step_size]
        else:
            break

    print(f"prepared test data with shape: {complete_pattern.shape}")

    return complete_pattern


def cut_clock_in_pieces(data, technical_details):
    """
    This method cuts the clock into pieces similar to the processing of the training data.
    """
    step_size = technical_details["sample_freq"] // technical_details["clock_freq"]
    count_cuts = technical_details["signal_length"] // step_size
    steps_to_left = technical_details["steps_to_left"]
    steps_to_right = technical_details["steps_to_right"]

    # Adjust symbols_start_at_sample to the closest zero crossing from negative to positive in the clock
    clock = find_clock(data, technical_details["clock_freq"], technical_details["sample_freq"], technical_details["signal_length"])
    zero_crossings = np.where(np.diff(np.signbit(clock)))[0]
    start_cut = zero_crossings[0] + 1
    if np.signbit(clock[start_cut + 1]):
        start_cut = zero_crossings[1] + 1

    start_cut = start_cut % 100
    clock = clock[start_cut:]

    if technical_details["do_normalize_data"]:
        clock = normalized_1d_array(clock)

    # Create array for the 5 symbol pieces
    res_pattern = []

    # First, this checks if the cut will be fully contained in the data left
    # If successful, cut a 5 symbol long piece out of the test data and repeat entire process with a shift of 1 symbol
    for i in range(steps_to_left + 1, count_cuts - steps_to_right):
        if (i + steps_to_right) * step_size <= len(data):
            seq = clock[(i - steps_to_left) * step_size:(i + steps_to_right) * step_size]
            res_pattern.append(seq)
        else:
            break

    print(f"prepared clock data with shape:{np.array(res_pattern).shape}")

    complete_pattern = np.array(res_pattern)

    return complete_pattern


def show_result_time(dictonary, technical_details, label="Test accuracy"):
    """
    This method plots the end result for the main_time script.
    """
    # Rescaling
    test_accuracy = np.array(list(dictonary.values())) * 100
    sample_freq = technical_details["sample_freq"]

    # Convert time shifts to nano seconds
    time_step_plot = sample_freq / 10 ** 9
    time_shifts = list(dictonary.keys())
    time_shifts = [float(i) for i in time_shifts]
    time_shifts = np.array(time_shifts) / time_step_plot
    min_time = np.amin(time_shifts)
    max_time = np.amax(time_shifts)

    # Adds the 25% accuracy line
    guessing_x = np.arange(min_time - 0.1 * time_step_plot, max_time + 0.1 * time_step_plot, 0.1)
    guessing_y = np.full(guessing_x.shape, 25)

    # Set the grid
    fig, ax = plt.subplots()
    plt.ylim([0, 103])
    plt.xlim([min_time - 0.1 * time_step_plot, max_time + 0.1 * time_step_plot])

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.grid(which='major', color='gray', linestyle="-")
    ax.grid(which='minor', color='#CCCCCC', linestyle="-")

    ax.plot(time_shifts, test_accuracy, 'x-', label=label, markersize=10)
    ax.plot(guessing_x, guessing_y, 'r')
    ax.legend(loc="lower right")
    plt.ylabel("Accuracy in %")
    plt.xlabel("Time in ns")
    plt.show()


def plot_header_signal(training_data, clock, rectangle):
    """
    This method plots the header in the probe signal during debugging.
    """
    training_data = training_data / np.max(training_data)
    clock = clock / np.max(clock) * 0.5
    rectangle = rectangle / np.max(rectangle)

    plt.figure()
    plt.plot(training_data, label="training data")
    plt.plot(clock, label="clock")
    plt.plot(rectangle, label="header model")
    plt.legend(loc='upper center')
    plt.show()


def plot_header_laser_driver(laser_driver, rectangle):
    """
    This method plots the header in the laser driver during debugging.
    """
    laser_driver = laser_driver / np.max(laser_driver)
    rectangle = rectangle / np.max(rectangle)

    plt.figure()
    plt.plot(laser_driver, label="laser driver")
    plt.plot(rectangle, label="header model")
    plt.legend(loc='upper center')
    plt.show()


def make_unique_file_name(path: str) -> str:
    i = 0
    rest, extension = os.path.splitext(path)
    while os.path.exists(path):
        i += 1
        path = f"{rest}_renamed{i}{extension}"

    return path


def write_result_dict(str_double_dict,
                      target_file_path="result.json", override=False, ):
    """
    This method saves the result to a csv file.
    """
    try:
        if not override and os.path.exists(target_file_path):
            warnings.warn(f"Result file exists at {os.path.abspath(target_file_path)}.")
            target_file_path = make_unique_file_name(target_file_path)
            warnings.warn(f"Changed result file path to {target_file_path}.")

        git_hash, git_diff = get_repo_sha1_and_diff(search_parent_directories=True)
        text_git_diff = [str(d) for d in git_diff]

        str_double_dict["GIT_SHA1_ONSAVING"] = git_hash
        str_double_dict["GIT_DIFF_ONSAVING"] = text_git_diff

        with open(os.path.join(target_file_path), 'w+') as file:
            json.dump(str_double_dict, file)
        print(f"created a json file of the test accuracies at {target_file_path}")
    except Exception as e:
        warnings.warn(f"Failed to create json file at {target_file_path}.\n Error: {e}")


def load_result_csv(result_csv_filepath):
    distance = []
    test_accuracy = []
    with open(result_csv_filepath, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            distance.append(float(row[0]))
            test_accuracy.append(float(row[1]))
    print(distance, test_accuracy)
    return np.array(distance), np.array(test_accuracy)


def load_result(filepath="result.csv"):
    """
    This method loads the previous result from a csv file.
    """
    assert os.path.exists(filepath), f"Result file {filepath} does not exist."

    if filepath.endswith('.csv'):
        return load_result_csv(filepath)
    elif filepath.endswith('.json'):
        with open(filepath, "r") as f:
            res = json.load(f)
        return res
    else:
        raise ValueError(f"Unexpected file extension for result file {filepath}")


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

