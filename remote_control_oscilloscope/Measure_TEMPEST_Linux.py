import time
import os
import shutil
import datetime

import numpy as np

from osci import Oscilloscope
import verify_dataset_hash
from oldToNewFormat import convert_key_file
from remote_control_QKDSender import alice_control, ram_playback


def load_osci_csv(path):
    with open(path, 'r') as f:
        opened = np.genfromtxt(f, delimiter=',', skip_header=5)
    return opened


def measure_single_traces(
        oscilloscope: Oscilloscope,
        folder_path_on_osci: str = r"\\192.168.0.1\Users\LCRYADMIN\Documents\Emsec2d",
        dest_folder_path_on_PC: str = r"data_tmp",
        rename_file=lambda x: x,
):
    """

    :param rename_file: function to rename the files found on the oscilloscope when moving to PC folder
    :param oscilloscope:
    :param folder_path_on_osci:
    :param dest_folder_path_on_PC:
    :return: paths of files fetched from osci
    """
    # this also checks that the folder exists.
    assert len(os.listdir(folder_path_on_osci)) == 0, f"Clear target folder (oscilloscope) `{folder_path_on_osci}`:\n" \
                                                      f"`{os.listdir(folder_path_on_osci)}`)"

    print("measuring...")
    oscilloscope.store_traces_to_file_default_settings()  # actual measurement!
    print("...")
    time.sleep(3)  # wait for measurement to finish and files to be saved. TODO is this needed?????????????????????
    print("Measurement Done! Fetching result files.")

    measured_files = os.listdir(folder_path_on_osci)
    print(f"Measured files: {measured_files}")
    assert len(measured_files) != 0, "No measurement files found on oscilloscope. (Check path set on osci)"

    files_destinations = []
    for osci_file_name in measured_files:
        file_destination = os.path.join(dest_folder_path_on_PC, rename_file(osci_file_name))
        assert not os.path.exists(file_destination), f"Destination path already exists! {file_destination}"
        print(f"Moving file {osci_file_name} to destination {file_destination}.")
        shutil.move(os.path.join(folder_path_on_osci, osci_file_name), file_destination)
        assert os.path.exists(file_destination), "Copying files from osci failed! Moved file should exist but doesn't!"
        files_destinations.append(file_destination)

    time.sleep(0.5)  # TODO is this needed?????????????????????

    return files_destinations


def measure_train_validation_test(
        folder_path_PC,
        folder_path_osci,
        train_labels_path,
        test_labels_path,
        confirm_test_dataset=False,
        train_suffixes=None,
        test_suffixes=None,
):
    """make ram playback play training file. Copy it into the folder. Perform measurement."""
    if train_suffixes is None:
        train_suffixes = [1]

    if test_suffixes is None:
        test_suffixes = [1, 2]

    def map_file_name(prefix, osci_file_name, suffix_):
        if osci_file_name == "C1adomas00000.txt":
            return f"{prefix}_{suffix_}_clock_debug_signal.txt"
        elif osci_file_name == "C2adomas00000.txt":
            return f"{prefix}_{suffix_}_laser_driver_should_not_be_connected.txt"
        elif osci_file_name == "C3adomas00000.txt":
            return f"{prefix}_{suffix_}_data.txt"
        elif osci_file_name == "C4adomas00000.txt":
            return f"{prefix}_{suffix_}_synch_debug_signal.txt"
        else:
            raise NameError(f"No renaming specified for file name {osci_file_name}.")

    print(ram_playback(file_path=train_labels_path))  # sets the key repeated by Alice board
    # this also checks that the folder exists.
    assert len(os.listdir(folder_path_PC)) == 0, f"Clear target folder (PC) `{folder_path_PC}`:\n" \
                                                 f"`{os.listdir(folder_path_PC)}`)"
    shutil.copy(train_labels_path + '.txt', folder_path_PC)
    shutil.copy(train_labels_path + '.key', folder_path_PC)

    for suffix in train_suffixes:
        print(f"Measuring training dataset {suffix}...")
        result_files_train = measure_single_traces(osci, dest_folder_path_on_PC=folder_path_PC,
                                                   folder_path_on_osci=folder_path_osci,
                                                   rename_file=lambda name: map_file_name("train", name, suffix), )
        print(f"Training measurement '{suffix}' Done."
              f"Result files:\n{result_files_train}\n")

    print("---------- All training measurements Done! ---------------------------\n")
    if confirm_test_dataset:
        assert input(f"Prepare for test measurement and press enter to continue!") == ""
    else:
        print("Measuring test dataset...")

    print(ram_playback(file_path=test_labels_path))  # sets the key repeated by Alice board
    shutil.copy(test_labels_path + '.txt', folder_path_PC)
    shutil.copy(test_labels_path + '.key', folder_path_PC)
    time.sleep(1)  # wait for the folder to be empty again when files are moved!

    for suffix in test_suffixes:
        print(f"Measuring test dataset {suffix}...")
        result_files_test = measure_single_traces(osci, dest_folder_path_on_PC=folder_path_PC,
                                                  folder_path_on_osci=folder_path_osci,
                                                  rename_file=lambda name: map_file_name("test", name, suffix), )
        print(f"Test measurement '{suffix}' Done."
              f"Result files:\n{result_files_test}\n")

    print("---------- All test measurements Done! ---------------------------\n\n")


def measure(
        *,
        oscilloscope: Oscilloscope,
        folder_path_on_pc: str,
        train_suffixes,
        test_suffixes,
        laser_driver_enabled,
        train_key_file_path,
        test_key_file_path,
        folder_path_on_osci: str = os.path.join("LCRYADMIN", "Documents", "Emsec2d"),
        script_path_alice_control=r"/home/alice/git/pc-controller/cmake-build-release/bin/alice-control",
        script_path_file_playback=r"/home/alice/git/pc-controller/cmake-build-release/bin/ram-playback",
):
    if not os.path.exists(folder_path_on_pc):
        os.makedirs(folder_path_on_pc)

    # check access to osci
    assert oscilloscope.get_ID()
    folder_path_on_osci = os.path.join(f"/media/lecroy", folder_path_on_osci)
    assert len(os.listdir(folder_path_on_pc)) == 0, f"Clear folder PC {folder_path_on_pc}!"
    assert len(os.listdir(folder_path_on_osci)) == 0, f"Clear folder osci {folder_path_on_osci}!"

    # and to alice board
    assert os.path.isfile(script_path_alice_control)
    assert os.path.isfile(script_path_file_playback)
    print(40 * "-")
    if laser_driver_enabled:
        alice_control_args = "-c -1 -ms 255 -b 1 -da 100 -db 150"
    else:
        alice_control_args = "-c -1 -ms 0 -b 0 -da 100 -db 150"

    print(f"Laser driver enabled: {laser_driver_enabled}")
    print(alice_control(args=alice_control_args))
    print(40 * "-")
    assert input("Please confirm USB connection is successful!") == "", \
        "User input not empty. Assuming measurement was not confirmed!"

    # Do actual measurement!!!!!!!!!!!!!!!!!!!!!!
    x_ticks = list(range(0, 11))
    y_ticks = list(range(0, 11))

    meas_locations = [(x, y) for x in x_ticks for y in y_ticks]

    for meas_loc in meas_locations:
        try:
            tmp = input(f"Measuring new training data at {meas_loc}! Please confirm by `enter`. "
                        f"If input non-empty, it will be used as measurement folder name.\n"
                        f"(Except `END_MEAS` to end measurement)")
            if tmp == "END_MEAS":
                break
            elif tmp != "":
                measurement_label = tmp
            else:
                measurement_label = f"2d_{meas_loc[0]:.3f}_{meas_loc[1]:.3f}"
        except Exception as e:
            print(f"Ending measurement after keyboard input failed ({e}).")
            break

        current_folder_pc = os.path.join(folder_path_on_pc, measurement_label)
        os.makedirs(current_folder_pc)
        measure_train_validation_test(
            current_folder_pc, folder_path_on_osci,
            train_labels_path=train_key_file_path,
            train_suffixes=train_suffixes,
            test_labels_path=test_key_file_path,
            test_suffixes=test_suffixes,
        )

    # only write metadata if measurement was completed successfully
    metadata = {
        "alice_control_args": alice_control_args,
        "laser_driver_enabled": laser_driver_enabled,
        "time_measurement_end": datetime.datetime.now().strftime("%Y%m%d-%H:%M"),
        "evaluation_target": input("evaluation_target"),
        "probe": input("probe"),
        "amplifier": input("amplifier"),
        "comments": input("comments"),
    }
    return metadata


def main(oscilloscope):
    # convert text files representing keys into binary files containing same information.
    train_key_file_path = "training_labels"
    test_key_file_path = "test_labels"
    convert_key_file(filepath=train_key_file_path + '.txt', output_path=train_key_file_path + '.key', )
    convert_key_file(filepath=test_key_file_path + '.txt', output_path=test_key_file_path + '.key', )

    print(oscilloscope.get_ID())
    measurement_name = input("Specify folder name for measurement (enter to use default)")
    if len(measurement_name) == 0:
        measurement_name = f"tmp_{datetime.datetime.now():%Y-%m-%d_%H_%M}"

    default_n_train_measurements = 2
    n_train_measurements = int(input(f"How many training measurements do you want for each position? "
                                     f"(Default {default_n_train_measurements})") or default_n_train_measurements)

    default_n_test_measurements = 1
    n_test_measurements = int(input(f"How many test measurements do you want for each position? "
                                    f"(Default {default_n_test_measurements})") or default_n_test_measurements)

    data_folder_path = os.path.join("..", "paper", "Data2D", measurement_name)
    laser_driver_enabled = True
    metadata = measure(
        oscilloscope=oscilloscope,
        folder_path_on_pc=data_folder_path,
        laser_driver_enabled=laser_driver_enabled,
        train_suffixes=list(range(n_train_measurements)),
        test_suffixes=list(range(n_test_measurements)),
        train_key_file_path=train_key_file_path,
        test_key_file_path=test_key_file_path
    )

    verify_dataset_hash.put_metadata_json_into_directory(data_folder_path, **metadata)


if __name__ == "__main__":
    osci = Oscilloscope(ip_addr="192.168.0.1")  # global variable! Can use in Python console for testing!

    main(osci)
