import win32api
import win32con
from config import vk


if __name__ == "__main__":
    import time
    import utils
    import pickle
    vk_keys = list(vk.keys())
    sender = utils.UDPserver(ip_addr="127.0.0.1", port=1224)
    while True:
        # 发送flags还是res？
        # 取决于场景，如果是通用场景自然是flags
        # 如果是专用场景直接发送res可以减少被控端的时间

        # 客户端代码

        flags = [win32api.GetKeyState(vk[each]) for each in vk_keys]
        res = [i for i, e in enumerate(flags) if e < 0]
        # data = pickle.dumps(flags)
        data = str(res).encode("utf-8")
        sender.send_data("127.0.0.1", data)
        print("发送给客户端 %s 的数据: %s\n" % ("127.0.0.1", data))
        # client_socket.close()
        # 
        # for each in res:
        #     print(f"Now {vk_keys[each]} is pressed.")