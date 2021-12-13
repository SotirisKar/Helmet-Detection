import os
import pathlib

path = '/home/sotiris/Downloads/labels_my-project-name_2021-11-22-03-11-56'
files = os.listdir(path)

for i in files:
    x = pathlib.Path(i).stem
    with open('{}/{}'.format(path, i), "r") as f:
        xml_str = f.read()
    xml_str = xml_str.replace(x+".jpg", x+"g12f.jpg")
    with open('{}/{}'.format(path, i), "w") as f:
        f.write(xml_str)
print('Conversion Complete.')
