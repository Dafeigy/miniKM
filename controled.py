from comm import UDPserver
# from utils import keys
from config import vk

import win32api
import win32con

from config import action_dict
# key = keys.Keys()
vk_keys = list(action_dict.keys())
listener = UDPserver(ip_addr="192.168.0.124", port = 1224)
last_data = []
while True:
    # print("Listening >>>")
    rec_data = eval(listener.receive_data())
    rel_data = [each for each in last_data if last_data not in rec_data]
    last_data = rec_data
    # Release
    for index in rel_data:
        win32api.keybd_event(vk[action_dict[vk_keys[index]]],0,win32con.KEYEVENTF_KEYUP,0)
    # Press
    for index in rec_data:
        win32api.keybd_event(vk[action_dict[vk_keys[index]]],0,0,0)