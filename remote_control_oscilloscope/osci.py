r"""
# Remote control the Lecry Osci (Wave Pro 604HD) from a Windows machine.
Linux works too, with some work (best mount the fils system).

Connect PC to Oscilloscope via Ethernet cable.
Set IP address and default gateway manually on both PC and Oscillocsope
For example, (at PC)   set IP of PC to 192.168.0.2 and default gateway to 192.168.0.1
             (at OSci) set IP of PC to 192.168.0.1 and default gateway to 192.168.0.2
Verify connection, e.g., write \\192.168.0.1 into explorer to access files on Osci. Have to know user+password

Make virtual environment, pip install
pyvisa
pyvisa-py

In Osci, in menu `Utilities/Setup/Remote`, set "Control from" to `LXI, (VXI11)`.

Check if it works by running
```python
import pyvisa
rm = pyvisa.ResourceManager()
osci = rm.open_resource('TCPIP0::192.168.0.1::inst0::INSTR')  # check ID of Osci in menu `Utilities/Setup/Remote`
osci.query("*IDN?")
```
you should get the device ID, containing the word "LECROY".

See (bottom part of)
https://cdn.teledynelecroy.com/files/manuals/maui-remote-control-and-automation-manual.pdf
for what commands can be used. Have fun!

Note: To get files from the device, you can just access them in the Windows file system.
```
import os
import shutil

osci_folder = r"\\192.168.0.1\Users\LCRYADMIN\Documents\Emsec2d"

os.listdir(osci_folder) # get files in folder

shutil.copy(os.path.join(osci_folder, 'C1F1C1C3signal00000000010000000000.txt'), ".")
```

(Most of these things are actually done by this script)

Before the measurement, make sure the Oscilloscope settings are correct.
We measure at sampling rate of 10GS/s, with 20us\div, for a total buffer size of 2MS.
These things are important to make sure that the header (for finding reference between training-key-file and measured
signal) is contained in the measurement. The header must be contained in the training-key-file.
 """

import pyvisa
# import usbtmc  # this is something like VISA, could also be used to control osci
import numpy as np
import time


class Oscilloscope:
    """
    Class to control Lecry Oscilloscope (Wave Pro 604HD) via Ethernet.
    For more commands, see (bottom part of)
    https://cdn.teledynelecroy.com/files/manuals/maui-remote-control-and-automation-manual.pdf
    """

    def __init__(self, ip_addr="192.168.0.1", min_number_aquired=1000, hist_channel="F1", timeout=10_000):
        self.min_number_aquired = min_number_aquired
        self.hist_channel = hist_channel
        self.timeout = timeout  # ms
        self.ip_address = ip_addr
        if self.__connect(ip_addr):
            print("connected")
        self.scope.timeout = self.timeout

    def __del__(self):
        self.scope.close()
        self.pyvisa_resource_manager.close()

    def __connect(self, ip):
        # connect to oscilloscope using VISA
        self.pyvisa_resource_manager = pyvisa.ResourceManager()
        self.scope = self.pyvisa_resource_manager.open_resource(f'TCPIP0::{ip}::inst0::INSTR')
        ID = self.get_ID()
        assert ID.find("LECROY") >= 0, f"Connection was not successful? ID returned from device: {ID}"
        return self.scope

    def _ask(self, query: str):
        return self.scope.query(query)

    def _write(self, command: str):
        self.scope.write(command)

    def wait_until_idle(self, for_n_seconds=1.0):
        return self._ask(f"""vbs? 'return=app.WaitUntilIdle({for_n_seconds})' """)

    def store_traces_to_file_default_settings(self):
        """
        Set the settings manually in the oscilloscope UI.
        See also `STORE_SETUP` command to configure some settings remotely, which is quite limited though.
        This is not thoroughly tested!
        """
        self._write(r"""vbs 'app.acquisition.triggermode = "stopped" ' """)
        self.wait_until_idle(for_n_seconds=0.5)
        self._write("STORE ALL_DISPLAYED,FILE")
        self.wait_until_idle(for_n_seconds=1.)
        self._write(r"""vbs 'app.acquisition.triggermode = "auto" ' """)

    def clear_sweeps(self):
        """clear sweeps of the histogram"""
        return self._write("CLEAR_SWEEPS")

    def get_ID(self) -> str:
        """Should return something like `*IDN LECROY,WP604HD,LCRY4607N02347,9.3.0`"""
        return self._ask("*IDN?")

    def wait_until_filled(self, hist_channel=None, min_number_aquired=None):
        """not properly tested, may contain errors."""
        if hist_channel is None:
            hist_channel = self.hist_channel
        if min_number_aquired is None:
            min_number_aquired = self.min_number_aquired
        # wait until the population is large enough
        number_acquired = 0
        while number_acquired < min_number_aquired:
            time.sleep(1)  # wait one second
            query = self._ask(f"{hist_channel}:PAVA? TOTP").split(",")
            if query[2][0:2] != "IV":  # not invalid value (insufficient data provided)
                number_acquired = float(query[1])
        return number_acquired

    def get_fwhm(self, hist_channel=None):
        """not properly tested, may contain errors."""
        if hist_channel is None:
            hist_channel = self.hist_channel

        # full width half max
        query = self._ask(f"{hist_channel}:PAVA? FWHM").split(",")

        return float(query[1].split()[0])

    def get_wavedesc(self, hist_channel=None):
        """not properly tested, may contain errors."""
        if hist_channel is None:
            hist_channel = self.hist_channel

        # get wave description
        return self._ask(f"{hist_channel}:INSPECT? 'WAVEDESC'")

    def get_time(self, hist_channel=None):
        """not properly tested, may contain errors."""
        if hist_channel is None:
            hist_channel = self.hist_channel

        # get time scaling
        query = self._ask(f"{hist_channel}:INSPECT? 'WAVE_ARRAY_COUNT'").split()
        data_length = float(query[3])
        query = self._ask(f"{hist_channel}:INSPECT? 'HORIZ_OFFSET'").split()
        horiz_offset = float(query[3])
        query = self._ask(f"{hist_channel}:INSPECT? 'HORIZ_INTERVAL'").split()
        horiz_interval = float(query[3])

        t = np.linspace(horiz_offset, horiz_offset + horiz_interval * data_length, data_length, endpoint=False)

        return t

    def get_hist_raw(self, hist_channel=None):
        """not properly tested, may contain errors."""
        if hist_channel is None:
            hist_channel = self.hist_channel

        # get raw data
        self._ask("COMM_FORMAT DEF9,WORD,BIN")
        d = self._ask_binary_values(f"{hist_channel}:WF? DAT1", datatype='h', container=np.array)  # TODO method missing

        # get raw data of histogram as displayed as two-byte integers (alternative)
        # self._ask(f"{hist_channel}:INSPECT? 'DATA_ARRAY_1',WORD")

        # get scaling
        query = self._ask(f"{hist_channel}:INSPECT? 'VERTICAL_GAIN'").split()
        vertical_gain = float(query[3])
        query = self._ask(f"{hist_channel}:INSPECT? 'VERTICAL_OFFSET'").split()
        vertical_offset = float(query[3])

        # apply scaling
        d = d * vertical_gain - vertical_offset

        # get time
        t = self.get_time(hist_channel=hist_channel)

        if len(t) != len(d):
            print("! time and data length does not match !")

        return t, d

    def get_hist(self, hist_channel=None):
        """not properly tested"""
        if hist_channel is None:
            hist_channel = self.hist_channel

        # get processed histogram as float
        query = self._ask(f"{hist_channel}:INSPECT? 'DATA_ARRAY_1',FLOAT").split('"')
        d = np.array(query[1].split(), float)

        # get time
        t = self.get_time(hist_channel=hist_channel)

        if len(t) != len(d):
            print("! time and data length does not match !")

        return t, d
