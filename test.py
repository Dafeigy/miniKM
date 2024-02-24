import torch
st = torch.load('model/demo1_best_model.pt')

torch.save(st.state_dict(),'new_pt.pt')
