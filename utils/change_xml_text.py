import os
import pathlib

path = '/home/sotiris/Downloads/new'
files = os.listdir(path)

for i in files:
    x = pathlib.Path(i).stem
    with open('{}/{}'.format(path, i), "r") as f:
        xml_str = f.read()
    xml_str = xml_str.replace('mororcycle', 'motorcycle')
    with open('{}/{}'.format(path, i), "w") as f:
        f.write(xml_str)
print('Conversion Complete.')