'''         Image processing and computer vision
              Alejandra Avendaño y Pedro Ruiz
               Electronic engineering students
              Pontificia Universidad Javeriana
                      Bogotá - 2020
'''

#Import libraries
import numpy as np
import cv2
import glob
import os
import json
from camera_model import *

if __name__ == '__main__':
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  #prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
  objp = np.zeros((7 * 7, 3), np.float32)
  objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

  #Arrays to store object points and image points from all the images.
  objpoints = []  # 3d point in real world space
  imgpoints = []  # 2d points in image plane.

  #Path of the ubication of the images
  path = '/home/pedroruiz54/Escritorio/pr/t4/WideAngle'
  path_file = os.path.join(path, '*.jpg')

  #Load all images in the folder
  images = glob.glob(path_file)

  for fname in images:
    img = cv2.imread(fname) #Load the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to gray scale
    #Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    #If found, add object points, image points (after refining them)
    if ret == True:
      objpoints.append(objp)
      corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
      imgpoints.append(corners2)
      #Draw and display the corners
      img = cv2.drawChessboardCorners(img, (7, 7), corners2, ret)
      cv2.imshow('img', img)
      cv2.waitKey(20)

  cv2.destroyAllWindows()
  #Calibration parameters of the camera
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

  #Ask for extrinsic parameters
  d = float(input("Ingrese la distancia "))
  altura = float(input("Ingrese el altura "))
  tilt = float(input("Ingrese el angulo tilt  "))
  pan = float(input("Ingrese el angulo pan "))

  #Save data in JSON file
  file_name = 'calibration.json'
  json_file = os.path.join(path, file_name)
  data = {
    'tilt': tilt,
    'pan': pan,
    'd' : d,
    'h' : altura,
    'K': mtx.tolist(),
    'distortion': dist.tolist()
  }
  with open(json_file, 'w') as fp:
    json.dump(data, fp, sort_keys=True, indent=1, ensure_ascii=False)

  #Load data from JSON file
  with open(json_file) as fp:
    json_data = json.load(fp)

  # extrinsics parameters
  R = set_rotation(json_data['tilt'], json_data['pan'], 0)
  t = np.array([0, -1 * json_data['d'], json_data['h']])

  # create camera
  camera = projective_camera(json_data['K'], int(2 * json_data['K'][0][2]), int(2 * json_data['K'][1][2]), R, t)

  #Create the cube
  square_3D = np.array([[0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0], [-0.5, 0.5, 0],
                         [0.5, 0.5, 1], [0.5, -0.5, 1], [-0.5, -0.5, 1], [-0.5, 0.5, 1]])

  square_2D = projective_camera_project(square_3D, camera)

  #Connect points with lines
  image_projective = 255 * np.ones(shape=[camera.height, camera.width, 3], dtype=np.uint8)
  cv2.line(image_projective, (square_2D[0][0], square_2D[0][1]), (square_2D[1][0], square_2D[1][1]), (0, 255, 255), 2)
  cv2.line(image_projective, (square_2D[1][0], square_2D[1][1]), (square_2D[2][0], square_2D[2][1]), (0, 255, 255), 2)
  cv2.line(image_projective, (square_2D[2][0], square_2D[2][1]), (square_2D[3][0], square_2D[3][1]), (0, 255, 255), 2)
  cv2.line(image_projective, (square_2D[3][0], square_2D[3][1]), (square_2D[0][0], square_2D[0][1]), (0, 255, 255), 2)

  cv2.line(image_projective, (square_2D[4][0], square_2D[4][1]), (square_2D[5][0], square_2D[5][1]), (0, 0, 255), 2)
  cv2.line(image_projective, (square_2D[5][0], square_2D[5][1]), (square_2D[6][0], square_2D[6][1]), (0, 0, 255), 2)
  cv2.line(image_projective, (square_2D[6][0], square_2D[6][1]), (square_2D[7][0], square_2D[7][1]), (0, 0, 255), 2)
  cv2.line(image_projective, (square_2D[7][0], square_2D[7][1]), (square_2D[4][0], square_2D[4][1]), (0, 0, 255), 2)

  cv2.line(image_projective, (square_2D[0][0], square_2D[0][1]), (square_2D[4][0], square_2D[4][1]), (200, 0, 0), 2)
  cv2.line(image_projective, (square_2D[1][0], square_2D[1][1]), (square_2D[5][0], square_2D[5][1]), (200, 0, 0), 2)
  cv2.line(image_projective, (square_2D[2][0], square_2D[2][1]), (square_2D[6][0], square_2D[6][1]), (200, 0, 0), 2)
  cv2.line(image_projective, (square_2D[3][0], square_2D[3][1]), (square_2D[7][0], square_2D[7][1]), (200, 0, 0), 2)

  #Show the generated image
  #cv2.imwrite('/home/pedroruiz54/Escritorio/pr/t4/WideAngle/ImageProjective1.png', image_projective)
  cv2.imshow("Image", image_projective)
  cv2.waitKey(0)
  cv2.destroyAllWindows()