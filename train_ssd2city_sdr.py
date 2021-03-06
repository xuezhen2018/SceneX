import sys
import argparse
import os
import random
import numpy as np
import time
from PIL import Image

from mlagents.envs.environment import UnityEnvironment
import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable

from dataset.cityscapes_dataset import cityscapesDataset
from dataset.ssd_dataset import ssdDataset
from model.mlp import MLP
from model.fcn8s_sourceonly import FCN8s_sourceonly
from model.vgg import VGG16
from utils import loss

UNITY_PATH = "./engine/scenex_trainer_city/unity_model_linux.x86_64"

# real-world dataset parameters (validation purpose)
REAL_INPUT_SIZE = "640,320"
REAL_DATA_DIR = "../dataset/"
REAL_LIST_PATH = "./dataset/cityscapes_list/train_val.txt"
REAL_BATCH_SIZE = 1

# synthetic dataset parameters (training purpose)
SYN_IMG_NUM = 200
SYN_LIST_PATH = "./dataset/ssd_list/train.txt"
SYN_DATA_DIR = "./snapshots/unity_images3/"
SYN_BATCH_SIZE = 4
SYN_INPUT_SIZE = "640, 320" 

# main model parameters
NUM_INPUTS = 7
NUM_OUTPUTS = 10
MAIN_LR = 1e-2
MAX_FRAMES = 40
STEP_EACH_FRAME = 1

# task model parameters
TASK_MODEL_NAME = "FCN8s"
NUM_CLASSES = 19
TASK_LR = 5e-4
TASK_NUM_STEPS = 1001
TASK_STOPS = 1001

# other parameters
RESUME = ""

color_encoding = [
        [152, 251, 152], # terrain, 9
        [244, 35, 232], # sidewalk, 1
        [128, 64, 128], # road, 0
        [0, 80, 100],    # train, 16
        [70, 70, 70],   # building, 2
        [190, 153, 153], # fence, 4
        [102, 102, 156], # wall, 3
        [153, 153, 153], # pole, 5
        [107, 142, 35], # vegetation. 8
        [0, 0, 230],    # motorcycle, 17
        [119, 11, 32],  # bicycle, 18
        [220, 20, 60],  # person, 11
        [255, 0, 0],     # rider, 12
        [250, 170, 30],  # traffic light, 6
        [220, 220, 0],   # traffic sign, 7
        [0, 0, 142],    # car, 13
        [0, 60, 100],   # bus, 15
        [0, 0, 70],     # truck, 14
        [0, 0, 0],      # void, 255
        [70, 130, 180]  # sky, 10
    ]

label_encoding = [9, 1, 0, 16, 2, 4, 3, 5, 8, 17, 18, 11, 12, 6, 7, 13, 15, 14, 255, 10]

def get_unity_envs():
    # check the python environment
    print("Python version: ", sys.version)
    if (sys.version_info[0]<3):
        raise Exception("ERROR: ML-Agents Toolkit requires Python 3")
        
    # set the unity environment
    env = UnityEnvironment(file_name=UNITY_PATH, base_port=5005)
    brain = env.brain_names[0]
    env.reset(train_mode=True)[brain]
    
    return env, brain

def get_arguments():
    parser = argparse.ArgumentParser(description="train_ssd2city_sdr")
    # real-world dataset parameters
    parser.add_argument("--real_input_size", type=str, default=REAL_INPUT_SIZE, help="comma-separated string with height and width of real images.")
    parser.add_argument("--real_data_dir", type=str, default=REAL_DATA_DIR, help="path to the directory containing the real-world dataset.")
    parser.add_argument("--real_list_path", type=str, default=REAL_LIST_PATH, help="path to the file listing the real-world images.")
    parser.add_argument("--real_batch_size", type=int, default=REAL_BATCH_SIZE, help="batch size for real-world dataset.")
    # synthetic dataset parameters
    parser.add_argument("--syn_img_num", type=int, default=SYN_IMG_NUM, help="number of images generated per batch.")
    parser.add_argument("--syn_data_dir", type=str, default=SYN_DATA_DIR, help="path to the directory containing the synthetic dataset.")
    parser.add_argument("--syn_list_path", type=str, default="./dataset/ssd_list/train.txt", help="path to the file listing the synthetic images.")
    parser.add_argument("--syn_batch_size", type=int, default=SYN_BATCH_SIZE, help="batch size for synthetic dataset.")
    parser.add_argument("--syn_input_size", type=str, default=SYN_INPUT_SIZE, help="comma-separated string with height and width of syn images.")
    # dataloader parameters
    parser.add_argument("--random-mirror", action="store_false", help="whether to randomly mirror the inputs during the training")
    parser.add_argument("--random-scale", action="store_true", help="whether to randomly scale the inputs during the training")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for multi-thred dataloading.")
    parser.add_argument("--iter_size", type=int, default=1, help="accumulate gradients for iter_size iterations.")
    parser.add_argument("--ignore_label", type=int, default=255, help="the index of label to be ignored during training.")
    # main model parameters
    parser.add_argument("--num_inputs", type=int, default=NUM_INPUTS, help="input dimensions of main model.")
    parser.add_argument("--num_outputs", type=int, default=NUM_OUTPUTS, help="output dimensions of main model.")
    parser.add_argument("--hidden_size", type=int, default=256, help="dimension of hidden size of main model.")
    parser.add_argument("--main_lr", type=float, default=MAIN_LR, help="learning rate of main model.")
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES, help="max number of frames to train main model.")
    parser.add_argument("--step_each_frame", type=int, default=STEP_EACH_FRAME, help="number of steps to train main model for each optimization.")
    # task model parameters
    parser.add_argument("--task_model_name", type=str, default=TASK_MODEL_NAME, help="name of task model.")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES, help="number of classes to predict.")
    parser.add_argument("--task_lr", type=float, default=TASK_LR, help="learning rate of task model.")
    parser.add_argument("--task_num_steps", type=int, default=TASK_NUM_STEPS, help="number of training steps for task model.")
    parser.add_argument("--task_stops", type=int, default=TASK_STOPS, help="number of steps to stop training.")
    parser.add_argument("--power", type=float, default=0.9, help="decay parameter to compute the learning rate.")
    # environment parameters
    parser.add_argument("--gpu", type=str, default="0", help="choose gpu device.")
    parser.add_argument("--random_seed", type=int, default=1234, help="random seed.")
    parser.add_argument("--snapshot_dir", type=str, default="./snapshots/", help="where to save snapshots.")
    parser.add_argument("--resume", type=str, default=RESUME, help="dir to reload the model.")
    return parser.parse_args()

def get_images_by_attributes(args, n_batch, env, brain, attribute_list):
    print("start generate batch {0:2d} images num {1:4d}".format(n_batch, args.syn_img_num))
    # light & camera
    light_intensity = 0.3#attribute_list[0]
    light_rotation_x = 0.5#attribute_list[1]
    light_rotation_y = 0.5#attribute_list[2]
    camera_prob = 0.5#attribute_list[3]
    camera_position_x = 0.5#attribute_list[4]
    camera_position_y = 0.5#attribute_list[5]
    camera_rotation_x = 0.5#attribute_list[6]
    camera_rotation_y = 0.5#attribute_list[7]
    
    # x_delta
    building_x_delta = attribute_list[0]
    fence_x_delta = attribute_list[1]
    tree_x_delta = attribute_list[2]
    motorcycle_x_delta = attribute_list[3]
    person_x_delta = attribute_list[4]
    hang_x_delta = attribute_list[5]
    car_x_delta = attribute_list[6]
    
    # interval
    building_prob = 0.5#attribute_list[15]
    building_interval = 0.5#attribute_list[16]
    fence_interval = 0.5#attribute_list[17]
    tree_interval = 0.5#attribute_list[18]
    motorcycle_interval = 0.5#attribute_list[19]
    person_interval = 0.5#attribute_list[20]
    hang_interval = 0.5#attribute_list[21]
    car_interval = 0.5#attribute_list[22]
    
    env.reset(train_mode=True)[brain]
    
    start_time = time.time()
    
    img_path = os.path.join(args.syn_data_dir, "batch_"+str(n_batch), "images")
    lbl_path = os.path.join(args.syn_data_dir, "batch_"+str(n_batch), "labels")
    color_path = os.path.join(args.syn_data_dir, "batch_"+str(n_batch), "colors")
    
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(lbl_path):
        os.makedirs(lbl_path)
    if not os.path.exists(color_path):
        os.makedirs(color_path)

    for i_img in range(args.syn_img_num):
        env_info = env.step([n_batch, args.syn_img_num, 5, 1, 0,
                             light_intensity, light_rotation_x, light_rotation_y, camera_prob, camera_position_x, #5-9
                             camera_position_y, camera_rotation_x, camera_rotation_y, building_prob, 0.5,  #10-14
                             building_x_delta, building_interval, 0.5, 0.7, fence_x_delta,  #15-19
                             fence_interval, 0.1, 0.2, 0.5, 0.5,  #20-24
                             0.7, tree_x_delta, tree_interval, 0.5, 0.7,  #25-29
                             motorcycle_x_delta, motorcycle_interval, 0.5, 0.5, 0.5,  #30-34
                             0.7, person_x_delta, person_interval, 0.7, 0.5,  #35-39
                             0.5, 0.7, hang_x_delta, hang_interval, 0.5,  #40-44
                             0.5, 0.5, 0.9, car_x_delta, car_interval,  #45-49
                             0.6, 0.4, 0.5, 0.5, 0.5   #50-54
                             ])[brain]
        
        if i_img % 10 == 0:
            continue
        
        img = np.asarray(env_info.visual_observations[0][0]*255.0, dtype=np.uint8)
        lbl_color = np.zeros_like(img, dtype=np.uint8)
        lbl_color[:,:] = color_encoding[-1]
        lbl = np.ones(img.shape[:2], dtype=np.uint8)*label_encoding[-1]
        
        for j in range(19):
            img1 = np.asarray(env_info.visual_observations[j*2+1][0]*255.0, dtype=np.uint8)
            img2 = np.asarray(env_info.visual_observations[j*2+2][0]*255.0, dtype=np.uint8)
            
            rest = (img1==0)&(img2==255)
            
            img_copy = img.copy()
            img_copy[rest] = 0
            
            mask = (img_copy == img2)
            mask = (mask[:,:,0] & mask[:,:,1] & mask[:,:,2])
            
            lbl_color[mask, :] = color_encoding[j]
            lbl[mask] = label_encoding[j]
        
        img = Image.fromarray(img)
        img.save(img_path + "/%d.png"%(i_img+1))
        
        lbl = Image.fromarray(lbl)
        lbl.save(lbl_path + "/%d_label.png"%(i_img+1))
        
        lbl_color = Image.fromarray(lbl_color)
        lbl_color.save(color_path + "/%d_color.png"%(i_img+1))

        if (i_img % 100 == 0) & (i_img!=0):
            print("{0:3d} images generated! time: {1:.1f}".format(i_img, time.time()-start_time))
        
    print("batch {0:2d} generated! time: {1:.2f}".format(n_batch, time.time()-start_time))

def get_validation_dataset(args):
    w, h = map(int, args.real_input_size.split(','))
    input_size = (w, h)
    val_set = cityscapesDataset(root=args.real_data_dir, list_path=args.real_list_path, 
                                max_iters=None, crop_size=input_size, 
                                scale=False, mirror=False, set="train")
    return val_set

def get_training_dataset(args, batch):
    w, h = map(int, args.syn_input_size.split(','))
    input_size = (w, h)
    train_set = ssdDataset(root=args.syn_data_dir, batch=batch, list_path=args.syn_list_path, 
                           max_iters=args.task_num_steps*args.iter_size*args.syn_batch_size,
                           crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror)
    return train_set

def train_task_model(train_loader, val_loader, task_model, task_optimizer, args, cuda):
    print("start training the task model")
    start_time = time.time()
    train_loader = enumerate(train_loader)
    
    iou_best = 0
    
    for i_iter in range(args.task_num_steps):
        loss_seg = 0
        
        task_optimizer.zero_grad()
        adjust_learning_rate(args, task_optimizer, i_iter)
        
        for sub_i in range(args.iter_size):
            _, batch = next(train_loader)
            images, labels, _ = batch
            images, labels = Variable(images), Variable(labels.long())
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            
            pred = task_model(images)
            
            #segmentation loss
            loss_seg = loss_calc(pred, labels, cuda)
            loss_seg /= args.iter_size
            loss_seg.backward()
        
        task_optimizer.step()
        elapsed_time = time.time() - start_time
        
        if i_iter % 100 == 0:
            print("iter = {0:5d}/{1:5d}, loss_seg = {2:.3f}, time = {3:.1f}".format(i_iter, args.task_num_steps, loss_seg.item(), elapsed_time))

        if (i_iter % 200 == 0) & (i_iter!=0):
            task_model.eval()
            reward = val_task_model(val_loader, task_model, args, cuda)
            task_model.train()
            print("current reward: {0:.2f}".format(reward))
            if reward > iou_best:
                iou_best = reward
    
    print("save model ...")
    torch.save(task_model.state_dict(), os.path.join(args.snapshot_dir, "task_model_best.pth"))
    print("final reward: {0:.2f}".format(iou_best))
    return iou_best

def val_task_model(loader, model, args, cuda):
    print("start validating the task model")
    hist = np.zeros((args.num_classes, args.num_classes))
    for index, batch in enumerate(loader):
        #if index % 100 == 0:
        #    print("%d processed"%index)
        
        images, labels, _ = batch
        images = Variable(images)
        if cuda:
            images = images.cuda()
        
        with torch.no_grad():
            pred = model(images)
        
        pred = pred.cpu().data[0].numpy()
        pred = pred.transpose(1,2,0)
        pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8)
        
        label = labels.cpu().data[0].numpy()
        
        hist += fast_hist(label.flatten(), pred.flatten(), args.num_classes)
    
    mIoUs = per_class_iu(hist)
    print('===> mIoU 19 class: ' + str(round(np.nanmean(mIoUs)*100, 2)))
    return (np.nanmean(mIoUs)*100)**2

def fast_hist(a, b, n):
    k = (a>=0) & (a<n)
    return np.bincount(n*a[k].astype(int)+b[k], minlength=n**2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist)/(hist.sum(1)+hist.sum(0)-np.diag(hist))

def compute_returns(rewards, start):
    returns = []
    returns.append(start)
    
    decay = 0.9
    
    for i in range(0, len(rewards)):
        value = decay*returns[i]+(1-decay)*rewards[i]
        returns.append(value)
    return returns[1:]

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1-float(iter)/max_iter)**power)

def adjust_learning_rate(args, optimizer, i_iter):
    lr = lr_poly(args.task_lr, i_iter, args.task_num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    
def loss_calc(pred, label, cuda):
    criterion = loss.CrossEntropy2d()
    if cuda:
        criterion = criterion.cuda()
    return criterion(pred, label)

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
    # get unity environment
    env, brain = get_unity_envs()
    
    # get arguments
    args = get_arguments()
    print(args)
    
    # set gpu environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.enabled = True
    cudnn.benchmark = True
    cuda = torch.cuda.is_available()
    
    # set random seed
    rn = set_seeds(args.random_seed, cuda)

    # make directory
    os.makedirs(args.snapshot_dir, exist_ok=True)

    # get validation dataset
    val_set = get_validation_dataset(args)
    print("len of test set: ", len(val_set))
    val_loader = data.DataLoader(val_set, batch_size=args.real_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # generate training list
    with open(args.syn_list_path, "w") as fp:
        for i in range(args.syn_img_num):
            if i%10!=0:
                fp.write(str(i+1)+'\n')

    # get main model
    main_model = MLP(args.num_inputs, args.num_outputs, args.hidden_size)
    if args.resume != "":
        main_model.load_state_dict(torch.load(args.resume))

    # get task model
    if args.task_model_name == "FCN8s":
        task_model = FCN8s_sourceonly(n_class=args.num_classes)
        vgg16 = VGG16(pretrained=True)
        task_model.copy_params_from_vgg16(vgg16)
    else:
        raise ValueError("Specified model name: FCN8s")

    # save initial task model
    torch.save(task_model.state_dict(), os.path.join(args.snapshot_dir, "task_model_init.pth"))

    if cuda:
        main_model = main_model.cuda()
        task_model = task_model.cuda()

    # get optimizer
    main_optimizer = optim.Adam(main_model.parameters(), lr=args.main_lr)
    task_optimizer = optim.SGD(task_model.parameters(), lr=args.task_lr, momentum=0.9, weight_decay=1e-4)

    frame_idx = 0
    whole_start_time = time.time()
    while frame_idx < args.max_frames:
        
        log_probs = []
        rewards = []

        start_time = time.time()
        
        for i_step in range(1, args.step_each_frame+1):
            
            # get initial attribute list
            state = np.random.rand(1, args.num_inputs)
            state = torch.from_numpy(state).float()
            
            if cuda:
                state = state.cuda()
            
            # get modified attribute list
            dist = main_model(state)
            action = dist.sample()

            action_actual = action.float() / 10.0   # [0, 0.9]
            
            # generate images by attribute list
            print("action: " + str(action_actual.cpu().numpy()))
            get_images_by_attributes(args, i_step, env, brain, action_actual[0].cpu().numpy())
            
            train_set = get_training_dataset(args, i_step)
            train_loader = data.DataLoader(train_set, batch_size=args.syn_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            
            # train the task model using synthetic dataset
            task_model.load_state_dict(torch.load(os.path.join(args.snapshot_dir, "task_model_init.pth")))
            
            reward = train_task_model(train_loader, val_loader, task_model, task_optimizer, args, cuda)
            log_prob = dist.log_prob(action)[0]
            
            log_probs.append(log_prob)
            rewards.append(torch.FloatTensor([reward]))
            
            frame_idx += 1

            if frame_idx == 1:
                moving_start = torch.FloatTensor([reward])
        
        baseline = compute_returns(rewards, moving_start)
        moving_start = baseline[-1]
            
        log_probs = torch.cat(log_probs)
        baseline = torch.cat(baseline).detach()
        rewards = torch.cat(rewards).detach()
    
        advantage = rewards - baseline
        if cuda:
            advantage = advantage.cuda()
        
        loss = -(log_probs*advantage.detach()).mean()
        
        with open(os.path.join(args.snapshot_dir, "logs.txt"), 'a') as fp:
            fp.write("frame idx: {0:4d}, state: {1:s}, action: {2:s}, reward: {3:s}, baseline: {4:s}, loss: {5:.2f} \n".format(
                    frame_idx, str(state.cpu()[0].numpy()), str(action.cpu()[0].numpy()), str(rewards.numpy()), str(baseline.numpy()), loss.item()))
        
        print("optimize the main model parameters")
        main_optimizer.zero_grad()
        loss.backward()
        main_optimizer.step()
        
        elapsed_time = time.time() - start_time
        print("[frame: {0:3d}], [loss: {1:.2f}], [time: {2:.1f}]".format(frame_idx, loss.item(), elapsed_time))
        
        torch.save(main_model.state_dict(), os.path.join(args.snapshot_dir, "main_model_%d.pth"%frame_idx))
    
    elapsed_time = time.time() - whole_start_time
    print("whole time: {0:.1f}".format(elapsed_time))
    env.close()
    
if __name__ == "__main__":
    main()