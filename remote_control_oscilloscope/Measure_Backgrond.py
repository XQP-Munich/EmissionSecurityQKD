import numpy as np
import time

import os
import shutil
from subprocess import check_output
import datetime

from osci import Oscilloscope
import verify_dataset_hash


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


def measure_background(
        folder_path_PC,
        folder_path_osci,
        suffixes=None,
):
    """Perform measurement."""
    if suffixes is None:
        suffixes = [1]

    def map_file_name(prefix, osci_file_name, suffix_):
        if osci_file_name == "C1adomas00000.txt":
            return f"{prefix}_{suffix_}_clock_debug_signal.txt"
        elif osci_file_name == "C2adomas00000.txt":
            return f"{prefix}_{suffix_}_laser_driver_should_not_be_connected.txt"
        elif osci_file_name == "C3adomas00000.txt":
            return f"{prefix}_{suffix_}.txt"
        elif osci_file_name == "C4adomas00000.txt":
            return f"{prefix}_{suffix_}_synch_debug_signal.txt"
        else:
            raise NameError(f"No renaming specified for file name {osci_file_name}.")

    # this also checks that the folder exists.
    assert len(os.listdir(folder_path_PC)) == 0, f"Clear target folder (PC) `{folder_path_PC}`:\n" \
                                                 f"`{os.listdir(folder_path_PC)}`)"

    for suffix in suffixes:
        print(f"Measuring background {suffix}...")
        result_files = measure_single_traces(osci, dest_folder_path_on_PC=folder_path_PC,
                                                   folder_path_on_osci=folder_path_osci,
                                                   rename_file=lambda name: map_file_name("background", name, suffix),)
        print(f"BAckground measurement '{suffix}' Done."
              f"Result files:\n{result_files}\n")

    print("---------- All background measurements Done! ---------------------------\n")


def measure(
        *,
        oscilloscope: Oscilloscope,
        folder_path_on_pc: str,
        suffixes,
        folder_path_on_osci: str = r"Users\LCRYADMIN\Documents\Emsec2d",
):
    if not os.path.exists(folder_path_on_pc):
        os.makedirs(folder_path_on_pc)

    # check access to osci
    assert oscilloscope.get_ID()
    folder_path_on_osci = os.path.join(f"\\\\{oscilloscope.ip_address}", folder_path_on_osci)
    assert len(os.listdir(folder_path_on_pc)) == 0, f"Clear folder PC {folder_path_on_pc}!"
    assert len(os.listdir(folder_path_on_osci)) == 0, f"Clear folder osci {folder_path_on_osci}!"

    try:
        tmp = input(f"Measuring background data! Please confirm by `enter`. "
                    f"If input non-empty, it will be used as measurement folder name.\n")
        if tmp != "":
            measurement_label = tmp
        else:
            measurement_label = f"Background"
    except Exception as e:
        print(f"Ending measurement after keyboard input failed ({e}).")

    current_folder_pc = os.path.join(folder_path_on_pc, measurement_label)
    os.makedirs(current_folder_pc)
    measure_background(
        current_folder_pc, folder_path_on_osci,
        suffixes=suffixes,
    )

    # only write metadata if measurement was completed successfully
    metadata = {
        "time_measurement_end": datetime.datetime.now().strftime("%Y%m%d-%H:%M"),
        "probe": input("probe"),
        "amplifier": input("amplifier"),
        "comments": input("comments"),
    }
    return metadata


def main(oscilloscope):
    print(oscilloscope.get_ID())
    measurement_name = input("Specify folder name for measurement (enter to use default)")
    if len(measurement_name) == 0:
        measurement_name = f"tmp_{datetime.datetime.now():%Y-%m-%d_%H_%M}"

    default_n_background_measurements = 10
    n_background_measurements = int(input(f"How many background measurements do you want for each position? "
                                     f"(Default {default_n_background_measurements})") or default_n_background_measurements)

    data_folder_path = os.path.join("..", "paper", "Data2D", measurement_name)
    metadata = measure(
        oscilloscope=oscilloscope,
        folder_path_on_pc=data_folder_path,
        suffixes=list(range(n_background_measurements)),
    )

    verify_dataset_hash.put_metadata_json_into_directory(data_folder_path, **metadata)


if __name__ == "__main__":
    osci = Oscilloscope(ip_addr="192.168.0.1")  # global variable! Can use in Python console for testing!

    main(osci)
