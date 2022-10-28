# ------------------------------------------------------------------------------
# Modified based on 
#   https://github.com/CommissarMa/MCNN-pytorch
#   https://github.com/WongKinYiu/yolov7
# ------------------------------------------------------------------------------
from ast import parse
from utils.utils import getClusterSubImages
from mcnn_model import MCNN
import cv2
import torch
import numpy as np
import os
import argparse

# yolo
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from models.experimental import attempt_load
from utils.datasets import letterbox

def loadYoloModel(weights_dir, device, half=True):
    model = attempt_load(weights_dir, map_location=device) # load FP32 model

    if half:
        model.half()  # to FP16

    # Warmup
    if device.type != 'cpu':
        if half:
            img = torch.rand((1, 3, 960, 544), device=device).half()
        else:
            img = torch.rand((1, 3, 960, 544), device=device)
        for i in range(3):
            model(img, augment=False)

    return model

def yolo_detect(origin_img, model, device, imgsz, stride, half=True):
    
    img = letterbox(origin_img, imgsz, stride=stride)[0] # Padded resize
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres=0.5)

    # Process detections
    if len(pred[0]):
        # Rescale boxes from img_size to origin_img size
        pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], origin_img.shape).round()

        return pred[0]

def mcnn_detect(mcnn, img, device):
    mcnn.eval()

    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_for_torch = img_RGB.transpose((2,0,1)) # convert to order (channel,rows,cols)
    img_tensor = torch.tensor(img_for_torch, dtype=torch.float).unsqueeze(0).to(device)
    et_dmap=mcnn(img_tensor)
    et_dmap=et_dmap.squeeze(0).squeeze(0).detach().cpu().numpy()
    
    dmap_uint8 = (et_dmap + abs(et_dmap.min())) / (et_dmap.max() + abs(et_dmap.min())) * 255
    dmap_uint8 = cv2.resize(dmap_uint8.astype(np.uint8), (img.shape[1], img.shape[0]))

    return dmap_uint8

def fusion_detect(origin_img, mcnn, yolo, device, half):
    dmap_uint8 = mcnn_detect(mcnn, origin_img, device)
    sub_imgs = getClusterSubImages(origin_img, dmap_uint8)

    stride = int(yolo.stride.max())  # model stride
    imgsz = check_img_size(origin_img.shape[1], s=stride)  # check img_size
    fusion_preds = torch.rand((0, 6), device=device)
    for sub_img, (x1, y1), (x2, y2) in sub_imgs:
        pred = yolo_detect(sub_img, yolo, device, imgsz, stride, half)
        if pred is not None:
            pred[:, 0] += x1
            pred[:, 1] += y1
            pred[:, 2] += x1
            pred[:, 3] += y1
            pred[:, :4] = xyxy2xywh(pred[:, :4])
            fusion_preds = torch.cat((fusion_preds, pred), 0)

    fusion_preds = fusion_preds.reshape((1, *fusion_preds.shape))
    fusion_preds = non_max_suppression(fusion_preds, conf_thres=0.5)
    return fusion_preds

def main(opt):
    origin_img = cv2.imread(opt.img_dir)

    device = torch.device(opt.device)
    mcnn_param_dir = opt.mcnn_param
    yolo_weights_dir = opt.yolo_weights

    # load model
    mcnn = MCNN().to(device)
    mcnn.load_state_dict(torch.load(mcnn_param_dir))
    yolo = loadYoloModel(yolo_weights_dir, device, opt.half)

    # fusion detect
    fusion_preds = fusion_detect(origin_img, mcnn, yolo, device, opt.half)
    if len(fusion_preds[0]):
        # show results
        for *xyxy, conf, cls in reversed(fusion_preds[0]):
            cv2.rectangle(origin_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 1)
    
    cv2.imshow("win", origin_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default="484.jpg", help='.jpg path')
    parser.add_argument('--mcnn_param', type=str, default='MCNN_weights/mcnn_marine_debris.param', help='mcnn .param path')
    parser.add_argument('--yolo_weights', type=str, default='Yolov7_weights/best.pt', help='yolo .pt path')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--half', action='store_true', help='float or double')
    opt = parser.parse_args()

    if opt.device == 'cpu':
        opt.half = False
    print(opt)
    main(opt)