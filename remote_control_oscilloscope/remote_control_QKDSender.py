from subprocess import check_output
import time
import os


def alice_control(script_path_alice_control=r"/home/alice/git/pc-controller/cmake-build-release/bin/alice-control",
                  args="-c -1 -ms 255 -b 1 -da 100 -db 150"):
    console_output = check_output(f"{script_path_alice_control} {args}", shell=True).decode()
    time.sleep(0.5)
    return console_output


def ram_playback(file_path,
                 script_path_file_playback=r"/home/alice/git/pc-controller/cmake-build-release/bin/ram-playback", ):
    actual_key_file_path = file_path + '.key'
    assert os.path.isfile(actual_key_file_path)
    console_output = check_output(f"{script_path_file_playback} -if {actual_key_file_path}", shell=True).decode()
    time.sleep(1.0)
    return console_output
