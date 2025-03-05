#!/usr/bin/env python
from argparse import ArgumentParser
import rospy
import numpy as np
import time
import torch
import mmcv
import os
import matplotlib.pyplot as plt
from PIL import Image as im
from tools.ros_utils import imgmsg_to_cv2, cv2_to_imgmsg
import time
import argparse
import math
from mmcv.parallel import MMDataParallel
from mmcv.runner import (load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
from sensor_msgs.msg import Image
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
import dataset.functional as Fd
from dataset.common import imagenet_mean, imagenet_std, colors_rugd, colors_city
from dataset.transforms import ToTensor
from utils.visualize import un_normalize
from mmseg.apis.inference import *
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
import cv2
import torch.nn.functional as F
from copy import deepcopy
import yaml
import rospy
from std_msgs.msg import Int32
from std_msgs.msg import String
import os


count = 0

# Global lists for navigable and non-navigable classes
Qn = []
Qnn = []
class_mapping = {}

# Load classes and mappings from the config file
def load_classes_from_config(config_file):
    global Qn, Qnn, class_mapping
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    Qn = config['navigable_classes']
    Qnn = config['non_navigable_classes']
    class_mapping = config['class_mapping']
    rospy.loginfo(f"Navigable Classes (Qn): {Qn}")
    rospy.loginfo(f"Non-navigable Classes (Qnn): {Qnn}")
    rospy.loginfo(f"Class Mapping: {class_mapping}")

def add_class_labels_to_image(image, seg_map, class_mapping, opacity=0.6):
    
    
    index_to_class = {v: k for k, v in class_mapping.items()}

    unique_classes = np.unique(seg_map)
    
    placed_labels = [] 

    for class_index in unique_classes:
        
        class_name = index_to_class.get(class_index, f"Class {class_index}")
        
        
        mask = (seg_map == class_index).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        largest_contour = max(contours, key=cv2.contourArea) if contours else None
        
        if largest_contour is not None:
          
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            else:
               
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_x, center_y = x + w // 2, y + h // 2
            
           
            if not any(abs(center_x - x) < 50 and abs(center_y - y) < 50 for x, y in placed_labels):
               
                cv2.putText(image, class_name, (center_x, center_y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
                            0.6, (255, 255, 255), 1, cv2.LINE_AA)
                placed_labels.append((center_x, center_y))  # Add to placed labels list
    
    return image

# Function to handle class update
def class_update_callback(msg):
    global Qn, Qnn
    # Split the incoming message into class_index and command (n/nn)
    try:
        class_info = msg.data.split(',')
        class_index = int(class_info[0].strip())  # Class index
        command = class_info[1].strip()  # Command: 'n' for navigable or 'nn' for non-navigable
    except (IndexError, ValueError):
        rospy.logwarn("Invalid message format. Use '<class_index>,<n_or_nn>'.")
        return

    if command == 'n':  # Make the class navigable
        if class_index in Qnn:  # If it's in non-navigable list, move it to navigable
            Qnn.remove(class_index)
            Qn.append(class_index)
            rospy.loginfo(f"Class {class_index} moved to navigable")
        elif class_index in Qn:
            rospy.loginfo(f"Class {class_index} is already navigable")
        else:
            rospy.logwarn(f"Class {class_index} not found in non-navigable list.")
    
    elif command == 'nn':  # Make the class non-navigable
        if class_index in Qn:  # If it's in navigable list, move it to non-navigable
            Qn.remove(class_index)
            Qnn.append(class_index)
            rospy.loginfo(f"Class {class_index} moved to non-navigable")
        elif class_index in Qnn:
            rospy.loginfo(f"Class {class_index} is already non-navigable")
        else:
            rospy.logwarn(f"Class {class_index} not found in navigable list.")
    else:
        rospy.logwarn(f"Invalid command '{command}'. Use 'n' for navigable or 'nn' for non-navigable.")



def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    if cfg.model.type == 'MultiResEncoderDecoder':
        cfg.model.type = 'HRDAEncoderDecoder'
    if cfg.model.decode_head.type == 'MultiResAttentionWrapper':
        cfg.model.decode_head.type = 'HRDAHead'
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg

def visualize_segmap(seg_map, name, dataset):
    # segmap : 1 x 21 x  h x w 
    #Qn=np.array([0,1,2,9,10,12,13,21,22])
    #Qnn=np.array([3,4,5,6,7,8,11,14,15,16,17,18,19,20,23])
    seg_map = seg_map.detach().cpu()
    
    if dataset == 'rugd':
        seg_map[seg_map == 255] = 24
    elif dataset == 'city':
        seg_map[seg_map == 255] = 19

    target = seg_map.argmax(1).squeeze()

    
    if dataset == 'rugd':
        colors_voc_origin = torch.Tensor(colors_rugd)
        new_im = colors_voc_origin[target.long()].numpy()
    elif dataset == 'city':
        colors_voc_origin = torch.Tensor(colors_city)
        new_im = colors_voc_origin[target.long()].numpy()
    new_im = new_im.astype(np.uint8)
    new_im = new_im[:, :, [2, 1, 0]]
    boundary=new_im.copy()
    
    for val in Qn:
        boundary[name[0] == val] = [255, 255, 255]

# Set pixels to (0, 0, 0) where the values are not in Qn
    for val in Qnn:
        boundary[name[0] == val] = [0,0,0]

    boundary_gray = cv2.cvtColor(boundary, cv2.COLOR_BGR2GRAY)
    last_black_pixels = np.zeros(boundary_gray.shape[1], dtype=int)
    for col in range(boundary_gray.shape[1]):
        column = boundary_gray[:, col]
        last_black_pixels[col] = np.where(column == 0)[0][-1]
    for col, last_black_pixel_row in enumerate(last_black_pixels):
        boundary_gray[0:last_black_pixel_row, col] = 0  # Set pixels above to black
        boundary_gray[last_black_pixel_row:, col] = 255  # Set pixels below to white

    
    boundary_updated = cv2.cvtColor(boundary_gray, cv2.COLOR_GRAY2BGR)
    
    return new_im, boundary, boundary_updated


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    #parser.add_argument('pretrained_ckpt', help='checkpoint file for eln')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--inference-mode',
        choices=['same', 'whole', 'slide'],
        default='same',
        help='Inference mode.')
    parser.add_argument(
        '--test-set',
        action='store_true',
        help='Run inference on the test set')
    parser.add_argument(
        '--hrda-out',
        choices=['', 'LR', 'HR', 'ATT'],
        default='',
        help='Extract LR and HR predictions from HRDA architecture.')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--dataset',
        type=str,
        default="city",
        help='dataset')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

class IDANAV_Manager(object):
    def __init__(self, model, palette, opacity):
        self.model = model
        self.model.eval()
        self.pal = palette
        self.opacity = opacity
        self.raw_image = None
        self.model_input = None
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        
        self.mean = np.float64(self.mean.reshape(1, -1))
        self.stdinv = 1 / np.float64(self.std.reshape(1, -1))
        
        self.pub_seg = rospy.Publisher('Multi_class_seg', Image, queue_size=10)
        self.pub_seg2 = rospy.Publisher('seg', Image, queue_size=10)
        self.pub_nav=rospy.Publisher('nav_image', Image, queue_size=10)
        self.pub_pre_nav=rospy.Publisher('pre_nav', Image, queue_size=10)
        #self.pub_nav2=rospy.Publisher('nav_image2', Image, queue_size=10)
        if args.dataset == 'rugd':
            num_classes = 24
        elif args.dataset == 'city':
            num_classes = 19
        else:
            raise ValueError
        rospy.Subscriber('/navigability', String, class_update_callback)

        # Load the initial configuration from YAML file
        load_classes_from_config('tools/navigability_config.yaml')#/home/khan/Downloads/Research_USA/seg_ws/DA
        
        rospy.Subscriber('/d400/color/image_raw', Image, self.callback)

        print('Initialization finished')

    def callback(self, msg):
        global count
        if count == 5:
           

            
           
            # The returned image is RGB
            self.raw_image = imgmsg_to_cv2(msg).astype(float)
            self.raw_image = cv2.resize(self.raw_image, (600, 338))

            self.original_raw_image = deepcopy(self.raw_image)
            
            # Normalize the RGB image using the mean and std
            cv2.subtract(self.raw_image, self.mean, self.raw_image)
            cv2.multiply(self.raw_image, self.stdinv, self.raw_image)  # inplace
            
            self.model_input = self.raw_image.transpose((2, 0, 1))
            

            

            self.model_input = torch.tensor(self.model_input).unsqueeze(0).to(torch.float32)

                  
            with torch.no_grad():
                result = self.model(return_loss=False, img=[self.model_input], img_metas=[None])
                name=result[0]
                name=np.array(name)
                ema_logits=result[1]
            
           
            
            self.model_input=self.model_input.detach()
            #self.model_input = self.model_input.to('cuda:0') 
            input_img = self.model_input.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
        
            
           
            pred_img, pre_bound, nav_bound=visualize_segmap(ema_logits,name, args.dataset)
            input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())  # Normalize to [0, 1]
            input_img = (input_img * 255).astype(np.uint8)  # Convert to 8-bit for OpenCV

            
            alpha = 0.6  
            beta = 1.0 - alpha

            blended_img = cv2.addWeighted(input_img, alpha, pred_img, beta, 0)
            blended_img_with_labels = add_class_labels_to_image(blended_img, name[0], class_mapping)
                        
            
          
            
            # If the self.pred_img is already in RGB format, then use 'rgb8' as the encoding
            self.pub_seg.publish(cv2_to_imgmsg(blended_img_with_labels, encoding='bgr8')) 
            self.pub_seg2.publish(cv2_to_imgmsg(pred_img, encoding='bgr8'))
            self.pub_nav.publish(cv2_to_imgmsg(nav_bound, encoding='8UC3')) 
            self.pub_pre_nav.publish(cv2_to_imgmsg(pre_bound, encoding='8UC3')) 

            #self.pub_nav2.publish(cv2_to_imgmsg(nav_bound, encoding='8UC3')) 
           
            count = 0
        else:
            count = count+1
        

if __name__ == '__main__':
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg = update_legacy_cfg(cfg)
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.inference_mode == 'same':
        # Use pre-defined inference mode
        pass
    elif args.inference_mode == 'whole':
        print('Force whole inference.')
        cfg.model.test_cfg.mode = 'whole'
    elif args.inference_mode == 'slide':
        print('Force slide inference.')
        cfg.model.test_cfg.mode = 'slide'
        crsize = cfg.data.train.get('sync_crop_size', cfg.crop_size)
        cfg.model.test_cfg.crop_size = crsize
        cfg.model.test_cfg.stride = [int(e / 2) for e in crsize]
        cfg.model.test_cfg.batched_slide = True
    else:
        raise NotImplementedError(args.inference_mode)

    if args.hrda_out == 'LR':
        cfg['model']['decode_head']['fixed_attention'] = 0.0
    elif args.hrda_out == 'HR':
        cfg['model']['decode_head']['fixed_attention'] = 1.0
    elif args.hrda_out == 'ATT':
        cfg['model']['decode_head']['debug_output_attention'] = True
    elif args.hrda_out == '':
        pass
    else:
        raise NotImplementedError(args.hrda_out)

    if args.test_set:
        for k in cfg.data.test:
            if isinstance(cfg.data.test[k], str):
                cfg.data.test[k] = cfg.data.test[k].replace('val', 'test')

    cfg.data.test['type'] = 'MESHDataset'
    cfg.data.test['data_root'] = 'MESH'

    if cfg.data.test['data_root'] is None:
        raise FileNotFoundError("DA directory not found.")
    
    dataset = build_dataset(cfg.data.test)
    
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')])
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE
    
    palette = model.PALETTE
    model = MMDataParallel(model, device_ids=[0])
        
    rospy.init_node('idanav_deploy', anonymous=True)
    
    my_node = IDANAV_Manager(model, palette, args.opacity)
    rospy.spin()
