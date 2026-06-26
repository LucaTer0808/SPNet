import time

import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import numpy as np
import os, argparse
import cv2
from Code.lib.model import SPNet
from Code.utils.data import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id',   type=str, default='0', help='select gpu id')
parser.add_argument('--rgb', type=str, default='RGB', help='rgb image path')
parser.add_argument('--depth', type=str, default='depth', help='depth image path')
parser.add_argument('--gt', type=str, default='GT', help='gt image path')
parser.add_argument('--target', type=str, default='target', help='path to save the predicted maps')

opt = parser.parse_args()

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
 

#load the model
model = SPNet(32,50)
model.cuda()

model.load_state_dict(torch.load('./Checkpoint/SPNet/SPNet_model_best.pth'))
model.eval()


if not os.path.exists(opt.target):
    os.makedirs(opt.target)
        
image_root  = opt.rgb if opt.rgb.endswith('/') else opt.rgb + '/'
gt_root     = opt.gt if opt.gt.endswith('/') else opt.gt + '/'
depth_root  = opt.depth if opt.depth.endswith('/') else opt.depth + '/'
target_path = opt.target if opt.target.endswith('/') else opt.target + '/'
eval_path = target_path + 'evaluation.txt'

test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)

torch.cuda.synchronize()
start_time = time.time()

for i in range(test_loader.size):
    print(f'Processing image {i+1}/{test_loader.size}')
    image, gt,depth, name, image_for_post = test_loader.load_data()
        
    gt      = np.asarray(gt, np.float32)
    gt     /= (gt.max() + 1e-8)
    image   = image.cuda()
    depth   = depth.cuda()
    pre_res = model(image,depth)
    res     = pre_res[2]     
    res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res     = res.sigmoid().data.cpu().numpy().squeeze()
    res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
    cv2.imwrite(target_path + name,res*255)

torch.cuda.synchronize()
end_time = time.time()

total_time = end_time - start_time
images_processed = test_loader.size
average_time_per_image = total_time / images_processed if images_processed > 0 else 0

with open(eval_path, 'w') as f:
    f.write(f'Total time: {total_time:.4f} seconds\n')
    f.write(f'Images processed: {images_processed}\n')
    f.write(f'Average time per image: {average_time_per_image:.4f} seconds\n')

print('Test Done!')
