names = {"H": 0b0000, "V": 0b0001, "P": 0b0100, "M": 0b0101}


def convert_key_file(filepath, output_path, debug=False):
    """Looks filepath (e.g. `training_labels.txt`), generates binary file (e.g. `training_labels.key`)"""
    symbols = []
    with open(filepath, "r") as inFile:
        for line in inFile.readlines()[2:]:
            line = line.strip("\n")
            i = 1
            if len(line) > 1:
                i = int(line[:-1])
            for _ in range(i):
                symbols.append(line[-1])
    with open(output_path, "wb") as outFile:
        # outFile.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')
        # outFile.write(len(symbols).to_bytes(4, "big"))
        for i in range(len(symbols) // 2):
            val = (names[symbols[2 * i]] << 4) + names[symbols[2 * i + 1]]
            if debug:
                print(symbols[2 * i], symbols[2 * i + 1], val)
            outFile.write(val.to_bytes(1, "big"))


if __name__ == "__main__":
    convert_key_file("training_labels.txt", output_path="training_labels.key")
    convert_key_file("test_labels.txt", output_path="test_labels.key")
