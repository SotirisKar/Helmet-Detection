from utils.display import displayMatrix
from multiprocessing import Process
import socket

s = socket.socket()
port = 8080
s.bind(('', port))
s.listen(1)
print('Socket is listening at port %s' % port)
image = ''
prev_msg = ''
n=h=nh = Process(target=displayMatrix, args=([image]))

while True:
    c, addr = s.accept()
    c.send('Connection successful'.encode())
    if 'msg' in locals():   # Keep memory of previous message
        prev_msg = msg
    msg = c.recv(2).decode()
    c.close()
    if prev_msg != msg:     # To avoid flickering & 2 images same time 
        if msg == 'H':
            image = '/home/pi/helmet-detection/images/helmet.jpg' # full path required
            h = Process(target=displayMatrix, args=([image]))
            if nh.is_alive() == True:
                nh.terminate()
                h.start()

        elif msg == 'NH':
            image = '/home/pi/helmet-detection/images/no_helmet.jpg' # full path required
            nh = Process(target=displayMatrix, args=([image]))
            if h.is_alive() == True:
                h.terminate()
            nh.start()

        elif msg == 'N':
            if h.is_alive() == True:
                h.terminate()
            if nh.is_alive() == True:
                nh.terminate()
