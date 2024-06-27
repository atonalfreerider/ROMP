import pickle
import torch

def convert_pkl_to_pytorch(input_path, output_path):
    with open(input_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    torch.save(data, output_path)

input_file = '/home/john/Desktop/3D-Pose/ROMP/simple_romp/smpl_model_data/SMPL_NEUTRAL.pkl'
output_file = '/home/john/Desktop/3D-Pose/ROMP/simple_romp/smpl_model_data/SMPL_NEUTRAL_pytorch.pkl'

convert_pkl_to_pytorch(input_file, output_file)
print(f"Converted {input_file} to {output_file}")
