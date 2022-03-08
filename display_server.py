from utils.display import displayMatrix
from multiprocessing import Process
import socket


os.chdir(os.path.dirname(os.path.abspath(__file__))
s = socket.socket()
port = 8080
s.bind(("", port))
s.listen(1)
print(f"Socket is listening at port {port}")
image = ""
prev_msg = ""
n=h=nh = Process(target=displayMatrix, args=([image]))

while True:
    c, addr = s.accept()
    print(f"Connection established with {addr}")
    c.send("Connection established".encode())
    if "msg" in locals():                                            # keep memory of previous message
        prev_msg = msg
    msg = c.recv(2).decode()
    c.close()
    if prev_msg != msg:                                              # to avoid flickering & 2 images at the same time 
        if msg == "H":
            image = "images/helmet.jpg"    # full path required
            h = Process(target=displayMatrix, args=([image]))
            if nh.is_alive() == True:
                nh.terminate()
                h.start()

        elif msg == "NH":
            image = "images/no_helmet.jpg" # full path required
            nh = Process(target=displayMatrix, args=([image]))
            if h.is_alive() == True:
                h.terminate()
            nh.start()

        elif msg == "N":                                             # stop displaying the image
            if h.is_alive() == True:
                h.terminate()
            if nh.is_alive() == True:
                nh.terminate()
