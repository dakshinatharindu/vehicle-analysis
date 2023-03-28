import torch
import cv2
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (check_img_size, Profile, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import (Annotator, colors)


REF = 0.55
BOTTOM_REF = 0.8


def run(source):
    global_min = 1080
    count = 0
    is_first_frame = True
    imgsz=(640, 640)

    model = DetectMultiBackend('weights/best.pt')
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    bs = 1
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, max_det=1000)
        
        for i, det in enumerate(pred):  # per image
            seen += 1

            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] 
            annotator = Annotator(im0, line_width=3, example=str(names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                if is_first_frame:
                    is_first_frame = False
                    h = im0.shape[0]
                    w = im0.shape[1]
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    vid_writer = cv2.VideoWriter("/tmp/result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                
                frame_min = h
                # print(h,w)
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    c = int(cls)
                    annotator.box_label(xyxy, None, color=colors(c, True))

                    y = int(xywh[1]*h)
                    if (y < frame_min and y > REF*h and y < BOTTOM_REF*h):
                        frame_min = y
                    if (y < global_min and y > REF*h and y < BOTTOM_REF*h):
                        count += 1
                global_min = frame_min
        
        im0 = annotator.result()
        cv2.line(im0, (0, int(REF*h)), (w, int(REF*h)), (0, 255, 0), 3)
        cv2.line(im0, (0, int(BOTTOM_REF*h)), (w, int(BOTTOM_REF*h)), (255, 0, 0), 3)
        cv2.putText(im0,'Detected Vehicles: ' + str(count),(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0, 0xFF, 0),2,cv2.FONT_HERSHEY_COMPLEX_SMALL)

        vid_writer.write(im0)

    vid_writer.release() 

    return count


