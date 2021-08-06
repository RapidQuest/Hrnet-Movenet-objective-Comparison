#Loader
import os
import cv2
import numpy as np
import json
import pandas as pd
import torch
import tensorflow as tf

class Loader:
  '''
  This Loader class upon instantiation returns iterator which returns processed tensors upon iteration for hrnet and movenet 
  model to directly feed into model as 
  well as generates target labels for comparison. 
  '''
  
  def __init__(self,path_to_test_set_directory,batch_size,cfg,movenet_interpreter):
    self.path_to_test_set_directory = path_to_test_set_directory
    self.batch_size = batch_size
    self.cfg
    self.coco_instance_category_names = coco_instance_category_names
  def get_len_of_iterations(self):
    self.list_of_images = os.listdir(self.path_to_test_set_directory+'/images')
    self.list_of_labels = os.listdir(self.path_to_test_set_directory+'/labels')
    return int(length/self.batch_size)
  def __iter__(self):
    self.len = self.get_len_of_iterations()
    self.index = 0
    return self
  def __next__():
    index = self.index
    if index == self.len:
      raise StopIteration
    self.index+=1
    return self.__get__item(index):
 
  def __get__item(self,index):
    list_of_image_batches = self.list_of_images[index*self.batch_size:(index+1)*self.batch_size]
    list_of_label_batches = self.list_of_labels[index*self.batch_size:(index+1)*self.batch_size]
    list_of_tuple = zip(list_of_image_batches,list_of_label_batches)
    batch_of_labels = []
    batch_of_movenet_tensor = []
    batch_of_hrnet_tensor = []
    batch_of_image_dimensions_for_movenet = []
    for i,j in list_of_tuple:
      image_bgr = cv2.imread(self.path_to_test_set_directory + '/' + i)
      batch_of_image_dimensions_for_movenet.append(np.swapaxes(np.array(image_bgr.shape[:-1]),0,1))
      image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
      box = self.detect_person(image)
      if box:
        continue
      hrnet_tensor = self.create_hrnet_tensor(image,box)
      batch_of_hrnet_tensor.append(hrnet_tensor)
      movenet_tensor = self.create_movenet_tensor(image,box)
      batch_of_movenet_tensor.append(movenet_tensor)
      annotation = self.get_json(j)
      label = self.get_label(self,annotation)
      list_of_labels.append(label)
    return torch.stack(batch_of_hrnet_tensor),np.stack(batch_of_movenet_tensor),np.stack(batch_of_labels),np.stack(batch_of_image_dimensions_for_movenet),{'center':self.center,'scale':self.scale}
  
  def detect_person(self,image,threhsold = 0.9):
    CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()
    input = []
    img_tensor = torch.from_numpy(image/255.).permute(2,0,1).float().to(CTX)
    input.append(img_tensor)
    output = boxmodel(input)
    pred_classes = np.array([self.coco_instance_category_names[i] for i in list(output[0]['labels'].cpu.numpy())])
    index = pred_classes=='person'
    pred_classes = pred_classes[index]
    pred_boxes = output[0]['boxes'].cpu.numpy()[index]
    pred_scores = output[0]['scores'].cpu.numpy()[index]
    if not pred_classes or np.max(pred_scores) < threshold:
      return False
    return pred_boxes[np.argmax(pred_scores)]


  def get_label(self,annotation):
    obj = pd.Series(annotation)
    label = np.stack(obj.to_numpy())
    return label
  
  def get_json(self,label_name):
    f= open(self.path_to_test_set_directory + '/' + label_name)
    return json.load(f)

  def create_hrnet_tensor(self,image,box):
   
      self.center, self.scale = box_to_center_scale(box, self.cfg.MODEL.IMAGE_SIZE[0], self.cfg.MODEL.IMAGE_SIZE[1])
      rotation = 0
      trans = get_affine_transform(self.center, self.scale, rotation, cfg.MODEL.IMAGE_SIZE)
      model_input = cv2.warpAffine(image,trans,(int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),flags=cv2.INTER_LINEAR)
      transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]), ])
      model_input = transform(model_input).unsqueeze(0)
      return model_input
  
  
  def create_movenet_tensor(self,image,box):
    bottom_left_corner = box[0]
    top_right_corner = box[1]
    cropped_image = image[int(bottom_left_corner[1]):int(top_right_corner[1]),int(bottom_left_corner[0]):int(top_right_corner[0])]
    input_image = tf.expand_dims(cropped_image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, 192,192) #this size applicable for movenet lightning
    input_image = tf.cast(input_image, dtype=tf.float32)
    return input_image.numpy()


    

  








