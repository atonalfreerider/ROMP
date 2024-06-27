import torch

def inspect_smpl(file_path):
    data = torch.load(file_path)
    print("Keys in the SMPL model:")
    for key in data.keys():
        print(f"- {key}")
    
    if 'extra_joints_index' not in data:
        print("\nThe 'extra_joints_index' key is missing.")
        print("Available keys that might be similar:")
        for key in data.keys():
            if 'joint' in key.lower() or 'index' in key.lower():
                print(f"- {key}")

file_path = '/home/john/Desktop/3D-Pose/ROMP/simple_romp/smpl_model_data/SMPL_NEUTRAL_pytorch.pkl'
inspect_smpl(file_path)
