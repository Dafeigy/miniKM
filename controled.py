from utils import UDPserver

listener = UDPserver(ip_addr="192.168.0.124", port = 1224)
while True:
    print("Listening >>>")
    listener.receive_data()