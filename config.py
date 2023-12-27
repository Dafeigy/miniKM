vk = {
    "A": 0x41,
    "D": 0x44,
    "E": 0x45,
    "F": 0x46,
    "H": 0x48,
    "I": 0x49,
    "J": 0x4A,
    "K": 0x4B,
    "L": 0x4C,
    "O": 0x4F,
    "S": 0x53,
    "U": 0x55,
    "W": 0x57,
    "Y": 0x59,

    "UP": 0x26,
    "LEFT": 0x25,
    "RIGHT": 0x27,
    "DOWN": 0x28,

    "ESC": 0x1B,
    "SPACE": 0x20, 
    "RETURN": 0x0D, "ENT": 0x0D,
}

nb_gNB_TX = 2

nb_UE_Ports = 1

nb_subcarrier = 624

nb_classes = 6

action_list = ['idle','punch','kick','left','right','down']
action_dict = {"idle":"SPACE",
               "punch":"I",
               "kick":"K",
               "left":"A",
               "right":"D",
               "down":"S",}