import torch
import tensorflow as tf
import Loader
from config import _C as cfg

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def parse_args():
  parser = argparse.ArgumentParser(description='hrnet-movenet comparision')
  parser.add_argument('--path', type=str, default=' ')
  parser.add_argument('--movenet_model_path', type=str, default=' ')
  args = parser.parse_args()
  return args

def main():
  cudnn.benchmark = cfg.CUDNN.BENCHMARK
  torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
  torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
  
  args = parse_args()


  
  
  #Instantiate hrnet model
  CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  hrnet_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)
  hrnet_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False) #If you are using cpu set device='cpu' in torch.load #gpu advisable
  hrnet_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
  hrnet_model.to(CTX)
  hrnet_model.eval()

  #Instantiate movenet model
  movenet = tf.lite.Interpreter(model_path=args.movenet_model_path)
  movenet.allocate_tensors()
  

  #Create loader 
  loader  = Loader(args.path,32,cfg,movenet)


  total_movenet_score = 0
  total_hrnet_score = 0
  for i,movenet_tensor,hrnet_tensor,label,frame_dimensions,post_process_hrnet_attributes in enumerate(loader):
    movenet.set_tensor(movenet.get_input_details()[0]['index'],movenet_tensor)
    movenet.invoke()
    output_movenet = movenet.get_tensor(movenet.get_output_details()[0]['index'])
    with torch.no_grad():
      output_hrnet = hrnet(hrnet_tensor)
    movenet_score = accuracy(output_movenet,movenet_label,'movenet',frame_batch = frame_dimensions)
    hrnet_score = accuracy(output_hrnet,hrnet_label,'hrnet',hrnet_post_process_attributes,config=cfg)
    total_movenet_score  = total_movenet_score + movenet_score
    total_hrnet_score = total_hrnet_score + hrnet_score
  average_movenet_score = total_movenet_score/(i+1)
  average_hrnet_score = total_hrnet_score/(i+1)
  relative_difference = (average_movenet_score - average_hrnet_score)/average_hrnet_score
  if relative_difference > 0:
    print("Movenet leads hr_net by {}%".format(abs(relative_difference)*100))
  else:
    print("Movenet lags hr_net by {}%".format(abs(relative_difference)*100))
  


if __name__ == '__main__':
    main()
  
  




  
