# ------------------------------------------------------------------------------
# Modified based on 
#   https://github.com/CommissarMa/MCNN-pytorch
#   https://github.com/WongKinYiu/yolov7
# ------------------------------------------------------------------------------
from utils.utils import getClusterSubImages
from mcnn_model import MCNN
import cv2
import torch
import numpy as np
import argparse
import os
import random

# yolo
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy
from models.experimental import attempt_load
from utils.datasets import letterbox, LoadImages
from utils.torch_utils import time_synchronized

def loadYoloModel(weights_dir, device, opt):
    model = attempt_load(weights_dir, map_location=device) # load FP32 model

    if opt.half:
        model.half()  # to FP16
        
    # Warmup
    if device.type != 'cpu':
        if opt.half:
            img = torch.rand((1, 3, 1920, 1088), device=device).half()
        else:
            img = torch.rand((1, 3, 1920, 1088), device=device)
        for _ in range(3):
            model(img, augment=False)

    return model

def yolo_detect(origin_img, model, device, imgsz, stride, opt):
    
    img = letterbox(origin_img, imgsz, stride=stride)[0] # Padded resize
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if opt.half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0][0]

    # Process detections
    if len(pred):
        # Rescale boxes from img_size to origin_img size
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], origin_img.shape).round()

        return pred
    return None

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

def fusion_detect(origin_img, mcnn, yolo, device, imgsz, stride, opt):
    t1 = time_synchronized()
    dmap_uint8 = mcnn_detect(mcnn, origin_img, device)
    t2 = time_synchronized()
    sub_imgs = getClusterSubImages(origin_img, dmap_uint8, opt)
    t3 = time_synchronized()

    fusion_preds = None#torch.rand((0, 6), device=device)
    for sub_img, (x1, y1), (_, _) in sub_imgs:
        pred = yolo_detect(sub_img, yolo, device, imgsz, stride, opt)
        if fusion_preds is None:
            fusion_preds = torch.rand((0, pred.size()[-1]), device=device)
        if pred is not None:
            pred[:, 0] += x1
            pred[:, 1] += y1
            pred[:, 2] += x1
            pred[:, 3] += y1
            pred[:, :4] = xyxy2xywh(pred[:, :4])
            fusion_preds = torch.cat((fusion_preds, pred), 0)

    pred = yolo_detect(origin_img, yolo, device, imgsz, stride, opt)
    if pred is not None:
        pred[:, :4] = xyxy2xywh(pred[:, :4])
        fusion_preds = torch.cat((fusion_preds, pred), 0)

    fusion_preds = fusion_preds.reshape((1, *fusion_preds.size()))
    fusion_preds = non_max_suppression(fusion_preds, conf_thres=opt.conf_thres)

    if opt.debug:
        print('Density detect:{:.1f}ms'.format((t2 - t1) * 1e3))
        print('Get clusters:{:.1f}ms'.format((t3 - t2) * 1e3))
        print('Yolo detect:{:.1f}ms'.format((time_synchronized() - t3) * 1e3))
        cv2.imshow('dmap', cv2.applyColorMap(dmap_uint8, cv2.COLORMAP_JET))
    return fusion_preds

def main(opt):
    device = torch.device(opt.device)
    mcnn_param_dir = opt.mcnn_param
    yolo_weights_dir = opt.yolo_weights

    # load model
    mcnn = MCNN().to(device)
    mcnn.load_state_dict(torch.load(mcnn_param_dir))    
    yolo = loadYoloModel(yolo_weights_dir, device, opt)

    stride = int(yolo.stride.max())  # model stride
    imgsz = check_img_size(opt.img_size, s=stride)  # check img_size
    dataset = LoadImages(opt.source, img_size=imgsz, stride=stride)
    vid_writer = vid_path = None

    # Get names and colors
    names = yolo.module.names if hasattr(yolo, 'module') else yolo.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Save file
    if opt.save_csv:
        csv = open(os.path.join(opt.save_path, 'result.csv'), 'w', newline='\n', encoding='utf-8')

    # fusion detect
    for path, _, im0, vid_cap in dataset:
        _, filename = os.path.split(path)

        # Predict
        t1 = time_synchronized()
        fusion_preds = fusion_detect(im0, mcnn, yolo, device, imgsz, stride, opt)
        t2 = time_synchronized()

        if len(fusion_preds[0]):
            # show results
            for *xyxy, conf, cls in reversed(fusion_preds[0]):
                label = f'{names[int(cls)]} {conf:.2f}'
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(im0, c1, c2, colors[int(cls)], 1) # Bounding box
                t_size = cv2.getTextSize(label, 0, fontScale=1 / 3, thickness=1)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(im0, c1, c2, colors[int(cls)], -1, cv2.LINE_AA)  # filled
                cv2.putText(im0, label, (c1[0], c1[1] - 2), 0, 1 / 3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                if opt.save_csv:  # Write to file
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    xywh_drone = [int(flt * 1920) if idx % 2 == 0 else int(flt * 1080) for idx, flt in enumerate(xywh)]
                    xywh_drone[0] = int(xywh_drone[0] - xywh_drone[2] / 2)
                    xywh_drone[1] = int(xywh_drone[1] - xywh_drone[3] / 2)
                    line = [filename.split('.')[0], str(int(cls)), ','.join([str(flt) for flt in xywh_drone])]
                    csv.write(','.join(line) + '\n')
        cv2.imshow(f"Predict Result", im0)
        
        # Save results (image with detections)
        if opt.save_img:
            save_path = os.path.join(opt.save_path, filename)
            if not os.path.exists(opt.save_path):
                os.makedirs(opt.save_path)
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {opt.save_path}")
            else:  # 'video'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

        print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference')
                    
        key = cv2.waitKey(0 if opt.debug else 1)
        if key == 27:
            break
    cv2.destroyAllWindows()
    if opt.save_csv:
        csv.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="484.jpg", help='source path')
    parser.add_argument('--mcnn-param', type=str, default='MCNN_weights/mcnn_marine_debris.param', help='mcnn .param path')
    parser.add_argument('--yolo-weights', type=str, default='Yolov7_weights/trash_best.pt', help='yolo .pt path')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--half', action='store_true', help='float or double')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--save-img', action='store_true', help='Save output')
    parser.add_argument('--save-csv', action='store_true', help='Save output as csv' )
    parser.add_argument('--save-path', type=str, default='output', help='save path of inference output')
    parser.add_argument('--debug', action='store_true', help='show middle results')
    opt = parser.parse_args()

    if opt.device == 'cpu':
        opt.half = False
    print(opt)
    main(opt)

    # python detect.py --source D:/Users/wbsc1/Downloads/public/ --half --img-size 1920 --save-csv --mcnn-param MCNN_weights/drone_best.param --yolo-weights Yolov7_weights/drone_e6e_best.pt --conf-thres 0.5