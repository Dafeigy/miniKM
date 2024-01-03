
import time
import ul_srs_est_pb2
import json
from comm import UDPserver
from google.protobuf import json_format
import FT
from config import action_list,action_dict

import Timesnet_DANN
import torch
import time

from model_utils import set_args

LISTEN_FROM = "192.168.0.19"
TARGET_IP = "192.168.0.124"
PORT_1 = 7776
PORT_2 = 1224
listener = UDPserver(ip_addr=LISTEN_FROM, port = PORT_1)
sender = UDPserver(ip_addr=LISTEN_FROM,port=PORT_2)

if __name__ == "__main__":
    start_time = time.time()
    device = torch.device('cuda:0')
    eval_mask = torch.ones(1,624).to(device)
    st = torch.load('best.pth')
    new_st = {key.replace("module.", ""): value for key, value in st.items()}
    opt=set_args()
    model = Timesnet_DANN.Model_domain(opt)
    model.load_state_dict(new_st)
    print("[ MODEL ] MODEL LOADED.]")
    model = model.to(device)
    model.eval()

    # WARMING UP MODEL
    for _ in range(100):
        print(f"[ MODEL ] WARMING UP : {_+1}/100 NOW:")
        st = time.time()
        random_data = torch.randn(1, 624, 1, 2).squeeze(0).permute(1,0,2).to(device)
        digits = model(random_data,eval_mask)
    print("[ MODEL ] WARMING UP FINISHED")
    print("="*120)
    print("Waiting for input coming ...")
    while True:
        st = time.time()
        ddd = ul_srs_est_pb2.NR_SRS_PACK()
        receive_data = listener.receive_data()
        ddd.ParseFromString(receive_data)
        sample=json.loads(json_format.MessageToJson(ddd))
        input_data = FT.rec_trans(sample).squeeze(0).permute(1,0,2).to(device)
        # sample shape : [1,624,1,2]
        bf_st = time.time()
        digits = model(input_data,eval_mask)
        output = torch.argmax(digits[0],dim=1)
        # signal_power = sample['SIGNALPOWER']             # int
        # noise_power = sample['NOISEPOWER']               # int
        print('【{:^10}】 {}'.format("Model",
                                    f"Model Predict as \033[5;37;42m【{action_list[int(output.cpu().numpy()[0])]}】\033[0m"))
        print('【{:^10}】 {}'.format("Model",
                                    f"digits tensor: {digits[0].tolist()[0]}"))
        print('【{:^10}】 {}'.format("System",
                                    f"From rec -> output using :{time.time() - st}s."))
        print('【{:^10}】 {}'.format("System",
                                    f"From process -> output using :{time.time() - bf_st}s."))
        

        # sender.send_data()
        res = [int(output.cpu().numpy()[0])]
        # data = pickle.dumps(flags)
        data = str(res).encode("utf-8")
        sender.send_data(TARGET_IP, data)
        print("Successfully send data.")