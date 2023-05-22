import os.path

import numpy.random as rand

path = "./test_labels.txt"
length = 15_102
header = False

print(os.path.abspath(path))

with open(path, 'wb') as file:
    formal_header = b"#Keyfile version:0.2 (do not change Header)\n#Use following Polarisations Signal (H,P,V,M), Decoy (h,p,v,m), Empty (e) seperate them with a new Line"
    file.write(formal_header)
    if header:
        file.write(b"\n100V")
    for i in range(length):
        x = rand.randint(0, 4)
        if x == 0:
            file.write(b"\nP")
        elif x == 1:
            file.write(b"\nH")
        elif x == 2:
            file.write(b"\nV")
        elif x == 3:
            file.write(b"\nM")
