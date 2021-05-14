import os
import time
import argparse
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from model.mlp_eval import MLP_eval

# main model parameters
NUM_INPUTS = 7
NUM_OUTPUTS = 10
MAX_FRAMES = 5
STEP_EACH_FRAME = 1

RESUME = "./snapshots/main_model_40.pth"


def get_arguments():
    parser = argparse.ArgumentParser(description="train_ssd_v1")
    # main model parameters
    parser.add_argument("--num_inputs", type=int, default=NUM_INPUTS, help="input dimensions of main model.")
    parser.add_argument("--num_outputs", type=int, default=NUM_OUTPUTS, help="output dimensions of main model.")
    parser.add_argument("--hidden_size", type=int, default=256, help="dimension of hidden size of main model.")
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES, help="max number of frames to train main model.")
    parser.add_argument("--step_each_frame", type=int, default=STEP_EACH_FRAME, help="number of steps to train main model for each optimization.")
    # environment parameters
    parser.add_argument("--random_seed", type=int, default=1234, help="random seed.")
    parser.add_argument("--snapshot_dir", type=str, default="./snapshots/", help="where to save snapshots.")
    parser.add_argument("--gpu", type=str, default="0", help="choose gpu device.")
    parser.add_argument("--resume", type=str, default=RESUME, help="dir to reload the model.")
    return parser.parse_args()

def set_seeds(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    
    return [
            np.random.rand(1),
            torch.randn(1)
        ]

def main():
    
    # get arguments
    args = get_arguments()
    print(args)
    
    # set the gpu environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.enabled = True
    cudnn.benchmark = True
    cuda = torch.cuda.is_available()

    rn = set_seeds(args.random_seed, cuda)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    
    # get main model
    main_model = MLP_eval(args.num_inputs, args.num_outputs, args.hidden_size)
    if args.resume != "":
        main_model.load_state_dict(torch.load(args.resume))
        
    #print(main_model.fc[0].weight)

    if cuda:
        main_model = main_model.cuda()

    # main training loop
    frame_idx = 0
    
    whole_start_time = time.time()
    while frame_idx < args.max_frames:
        
        start_time = time.time()
        
        for i_step in range(1, args.step_each_frame+1):
            
            # get initial attribute list
            state = np.random.rand(1, args.num_inputs)
            state = torch.from_numpy(state).float()

            if cuda:
                state = state.cuda()
            
            # get modified attribute list
            dist = main_model(state)
            action = torch.max(dist, dim=-1)[-1]
            
            action_actual = action.float() / 10.0   # [0, 0.9]

            print(action_actual)
            frame_idx += 1
        
        #with open(os.path.join(args.snapshot_dir, "logs.txt"), 'a') as fp:
        #    fp.write("frame idx: {0:4d}, action_reward: {1:s} \n".format(frame_idx, str(action_reward)))
        
        elapsed_time = time.time() - start_time
        print("[frame: {0:3d}], [time: {1:.1f}]".format(frame_idx, elapsed_time))
        
    elapsed_time = time.time() - whole_start_time
    print("whole time: {0:.1f}".format(elapsed_time))

if __name__ == "__main__":
    main()