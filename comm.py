import socket
import time

class UDPserver():
    def __init__(self, ip_addr, port:int) -> None:
        self.port = port
        self.IP = ip_addr
    
    def send_data(self, to_IP:str, data:str):
        # while True:
            # 1. 创建udp套接字
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
            # 2. 准备接收方的地址        
            dest_addr = (to_IP, self.port)  # 注意 是元组，ip是字符串，端口是数字
    
            
            # 4. 发送数据到指定的电脑上的指定程序中
            udp_socket.sendto(data, dest_addr)
            # print("发送给客户端 %s 的数据: %s\n" % (dest_addr, data))
    
            # 5. 关闭套接字
            udp_socket.close()

    def receive_data(self):
 
        # 1. 创建udp套接字
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
        # 2. 准备接收方的地址
        dest_addr = (self.IP, self.port)
    
        # 3. 绑定地址
        udp_socket.bind(dest_addr)
    
        # while True:
        # 4. 等待接收对方发送的数据
        receive_data, client_address = udp_socket.recvfrom(65535)
        # print("接收到了客户端 %s 传来的数据: %s\n" % (client_address, receive_data))
        return receive_data
        
if __name__ == "__main__":
    listener = UDPserver(port = 1224)
    while True:
        listener.receive_data()