from utils import UDPserver

listener = UDPserver(ip_addr="127.0.0.1", port = 1224)
while True:
    print("Listening >>>")
    listener.receive_data()