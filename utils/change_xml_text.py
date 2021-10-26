import os
import pathlib
files = os.listdir('/home/sotiris/Downloads/images')
for i in files:
  x = pathlib.Path(i).stem
  with open('/home/sotiris/Downloads/images/{}'.format(i)) as f:
      xml_str = f.read()
  xml_str = xml_str.replace(x+'.jpg', 's'+x+'.jpg')
  with open('/home/sotiris/Downloads/images/{}'.format(i), "w") as f:
      f.write(xml_str)
print('Done!')