#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import torch
import torchvision
import dlib
import cv2
from PIL import Image

points = {
  "jaw_points" : list(range(0, 17)),
  "r_braw_points" : list(range(17, 22)),
  "l_braw_points" : list(range(22, 27)),
  "nose_points" : list(range(27, 36)),
  "r_eye_points" : list(range(36, 42)),
  "l_eye_points" : list(range(42, 48)),
  "mouth_points" : list(range(48, 61)),
  "lips_points" : list(range(61, 68)),
  }

class FaceMask():

  def __init__(self, model_detection, model_landmarks, device='cpu', TH1=0.4, TH2=0.6):
    self.model_detection = model_detection
    self.model_landmarks = model_landmarks
    self.transforms = torchvision.transforms.PILToTensor()
    self.device = device
    self.TH1 = TH1
    self.TH2 = TH2

  def find_area_of_intersection(self, face, mask):
    x_max, y_max = min(mask[2], face[2]), min(mask[3], face[3])
    x_min, y_min = max(mask[0], face[0]), max(mask[1], face[1])
    if x_max < x_min or y_max < y_min:
      return 0
    return (x_max - x_min) * (y_max - y_min)

  def get_detection(self, picture):

    predictions = self.model_detection(picture.unsqueeze(0))
    faces = []
    masks = []
    for box, label, score in zip(predictions[0]["boxes"].tolist(),
                          predictions[0]["labels"].tolist(), 
                          predictions[0]["scores"].tolist()):

      if label == 2 and score >= self.TH2:
        faces.append(box)
      if label == 1 and score >= self.TH1:
        masks.append(box)

    return (faces, masks)

  def draw_something(self, boxes, labels, n, picture):
    draw_pic_with_rect(picture, boxes, labels, n)

  def face_mask_corr(self, faces, masks):
    face_mask = []
    for face in faces:
      max_area = 0
      best_mask = None
      for mask in masks:
        if self.find_area_of_intersection(face, mask) > max_area:
          max_area = self.find_area_of_intersection(face, mask)
          best_mask = mask
      face_mask.append(best_mask)
    return face_mask

  def find_landmarks(self, path, faces, masks):

    face_lands = []

    img = cv2.imread(path)

    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    face_lands = []
        
    for face in faces:
        x1 = int(face[0])
        y1 = int(face[1])
        x2 = int(face[2])
        y2 = int(face[3]) 

        box = dlib.rectangle(x1, y1, x2, y2)

        landmarks = self.model_landmarks(image=gray, box=box)
        # cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=1)

        el = {}
        el['lips_points'] = []
        el['nose_points'] = []

        for n in points['lips_points']:
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            el['lips_points'].append((x, y))


        for n in points['nose_points']:
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            el['nose_points'].append((x, y))
            # cv2.circle(img=img, center=(x, y), radius=1, color=(0, 255, 0), thickness=1)

        face_lands.append(el)
    return face_lands

    # for mask in masks:
    #     x1 = int(mask[0])
    #     y1 = int(mask[1])
    #     x2 = int(mask[2])
    #     y2 = int(mask[3]) 

    #     cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=1, )


    # cv2_imshow(img)   


  def faces_walker(self, path, faces, face_mask, face_lands, path_to_save):
    img = cv2.imread(path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for face, face_m, land in zip(faces, face_mask, face_lands):
        x1 = int(face[0])
        y1 = int(face[1])
        x2 = int(face[2])
        y2 = int(face[3]) 

        cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 139), thickness=2)

        if face_m is None:
            cv2.putText(img, 'No mask!!!', (x1, y1), font, 0.7, color=(0, 69, 255), thickness=2)
            continue

        mx1 = int(face_m[0])
        my1 = int(face_m[1])
        mx2 = int(face_m[2])
        my2 = int(face_m[3])

        cv2.rectangle(img=img, pt1=(mx1, my1), pt2=(mx2, my2), color=(255, 0, 0), thickness=2)
        
        out_of_lips = 0
        
        for point in land['lips_points']:
            x = point[0]
            y = point[1]
            if y > max(my1, my2) or y < min(my1, my2):
                out_of_lips += 1

        if (out_of_lips > 2):
            cv2.putText(img, 'Out of mouth', (x1, y1), font, 0.7, color=(0, 255, 0), 
                        thickness=2)
            continue

        out_of_nose = 0
        
        # for point in land['nose_points']:
        #     x = point[0]
        #     y = point[1]
        #     if y > max(my1, my2) or y < min(my1, my2):
        #         out_of_nose += 1

        # if (out_of_nose > 2):
        #     cv2.putText(img, 'Out of nose', (x1, y1), font, 0.7, color=(0, 69, 255), 
        #                 thickness=2)
        #     continue

        nose_y = [point[1] for point in land['nose_points']]
        nose_len = max(nose_y) - min(nose_y)
        nose_y1 = min(nose_y)
        nose_y2 = max(nose_y) 
        
        if nose_y2 < min(my1, my2):
            out_of_nose = nose_len
        elif nose_y1 > min(my1, my2):
            out_of_nose = 0
        else:
            out_of_nose = min(my1, my2) - nose_y1

        if (out_of_nose / nose_len > 0.4):
            cv2.putText(img, 'Out of nose', (x1, y1), font, 0.7, color=(0, 69, 255), 
                        thickness=2)
            continue
        
        cv2.putText(img, 'OK!!!', (x1, y1), font, 0.7, color=(50, 205, 155), 
                        thickness=2)

    cv2.imwrite(path_to_save, img)


  
  def __call__(self, path, path_to_save):
    picture_pil = Image.open(path)
    picture_torch = self.transforms(picture_pil).to(dtype=torch.float32)[:3, :, :]

    faces, masks = self.get_detection(picture_torch)

    face_mask = self.face_mask_corr(faces, masks)

    face_lands = self.find_landmarks(path, faces, masks)
    
    self.faces_walker(path, faces, face_mask, face_lands, path_to_save)

def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument("-img", dest="image_path",required=True)
	parser.add_argument(
		"-md", dest="model_detection_path", 
		type=str, required=False, default="../model_detection.pt")
	parser.add_argument(
		"-ml", dest="model_landmarks_path", 
		required=False, default="../model_landmarks.pt")
	parser.add_argument("-out", dest="out_image_path", 
		required=False, default="out.jpg")
	return parser.parse_args()


def main():
	# set up the parameters
	args = parse()
	current_path = os.getcwd()
	image_path = os.path.join(current_path, args.image_path)
	model_detection_path = os.path.join(current_path, args.model_detection_path)
	model_landmarks_path = os.path.join(current_path, args.model_landmarks_path)
	out_image_path = os.path.join(current_path, args.out_image_path)

	model_detection = torchvision.models.detection.fasterrcnn_resnet50_fpn()
	model_detection.load_state_dict(torch.load(model_detection_path, map_location=torch.device('cpu')))
	model_detection.eval()

	model_landmarks = dlib.shape_predictor(model_landmarks_path)

	find_mask = FaceMask(model_detection, model_landmarks)
	find_mask(image_path, out_image_path)


if __name__=="__main__":
	main()
