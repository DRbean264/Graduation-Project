import face_recognition
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import sys
import os
import cv2
import time
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def get_images_from_files(files_list,min_size,resize=False):
    """
    从文件读取图像，存放在一个列表中，元素为np.ndarray
    params:
    files_list:文件路径列表
    min_size:读入的图像最短边缩放到多长
    """
    images = []
    for f in files_list:
        image = cv2.imread(f)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if resize == True:
            scaled_shape = get_scaled_wh(image.shape[:2],min_size)
            image = cv2.resize(image,scaled_shape[::-1],interpolation=cv2.INTER_LINEAR)
        images.append(image)
    return images

def get_frames_through_idx(video_file,start,stop,min_size,resize=False):
    """
    从视频文件读取一定数量的帧，存放在一个列表中，元素为np.ndarray
    params:
    video_file:视频文件路径列表
    start:指标从1开始，开始的帧
    stop:结束的帧
    min_size:读入的图像最短边缩放到多长
    """
    video_capture = cv2.VideoCapture(video_file)
    
    frames = []
    count = 1
    while video_capture.isOpened():
        if count == stop + 1:
            break
        
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # 开始存储帧
        if count >= start:
#             frame = cv2.flip(frame, -1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize == True:
                scaled_shape = get_scaled_wh(frame.shape[:2],min_size)
                frame = cv2.resize(frame,scaled_shape[::-1],interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        
        count += 1
    video_capture.release()
    
    return frames
        
def yield_frames_through_idx(video_file,batch_size,min_size,resize=False):
    """
    从视频文件读取一定数量的帧，存放在一个列表中，元素为np.ndarray
    
    params:
    video_file:视频文件路径列表
    start:指标从1开始，开始的帧
    stop:结束的帧
    min_size:读入的图像最短边缩放到多长
    """
    video_capture = cv2.VideoCapture(video_file)

    while True:
        frames = []
        while video_capture.isOpened():
            for b in range(batch_size):
                ret, frame = video_capture.read()
                #  读到视频结尾的话，退出循环
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if resize == True:
                    scaled_shape = get_scaled_wh(frame.shape[:2],min_size)
                    frame = cv2.resize(frame,scaled_shape[::-1],interpolation=cv2.INTER_LINEAR)
                frames.append(frame)
            break
        if len(frames) != batch_size:
            video_capture.release()
    
        yield frames

def get_scaled_wh(current_size,min_size):
    """
    计算缩放后的宽高，计算方法为把短边缩放到指定的边长，等比缩放长边
    params:
    current_size:(h,w)
    min_size:最短的边长为多长
    
    return:
    等比例缩放后的高宽(h,w)
    """
    h_c,w_c = current_size
    
    scale_ratio = max(min_size/w_c,min_size/h_c)
    if h_c > w_c:
        h_s = int(h_c * scale_ratio)
        w_s = min_size
    else:
        h_s = min_size
        w_s = int(w_c * scale_ratio)
    return h_s,w_s

def get_location_landmark(images,number_of_times_to_upsample=0):
    """
    利用face_recognition库高速率实时的获取图像中人脸位置以及关键点位置，调用GPU运算
    params:
    images:一个列表，每个元素为一张图片的ndarray
    number_of_times_to_upsample：图像上采样次数，为了高效，取0
    """
    # return a list of tuples of found face locations in css (top, right, bottom, left) order
    # [[(72, 160, 140, 92), (24, 101, 71, 54), (17, 296, 57, 257), (48, 226, 95, 179), (1, 188, 41, 149), (1, 140, 41, 101)]]
    t0 = time.clock()
    batch_of_face_locations = face_recognition.batch_face_locations(images, number_of_times_to_upsample)
    t1 = time.clock()
    #print("Batch face locations and landmarks running time on {} image(s): {:.4f} s (Equivalent {:.2f} fps)" \
    #      .format(len(images),t1-t0,len(images)/(t1-t0)))
    
    batch_of_landmarks = []
    for idx,face_locations in enumerate(batch_of_face_locations):
        # 在这个应用中找到一个人脸说明正确
        if len(face_locations) == 1:
            # [{},{},{},...,{}]
            landmarks = face_recognition.face_landmarks(images[idx], face_locations=face_locations,model='large')
            batch_of_landmarks.append(landmarks)
            num_chin = len(landmarks[0]['chin'])
            num_eye = len(landmarks[0]['left_eye'])
        # 可能在这帧中找到多余的人脸，剔除小的人脸，这里假设最大的框是正确的
        elif len(face_locations) >= 2:
            idx_true = 0
            max_area = 0
            for id_face,face in enumerate(face_locations):
                top,right,bottom,left = face
                area = (bottom - top) * (right - left)
                if area > max_area:
                    max_area = area
                    idx_true = id_face
            landmarks = face_recognition.face_landmarks(images[idx], face_locations=face_locations[idx_true:idx_true+1],model='large')
            batch_of_landmarks.append(landmarks)
            num_chin = len(landmarks[0]['chin'])
            num_eye = len(landmarks[0]['left_eye'])
        # 也可能没找到人脸,直接返回，这个clip作废
        else:
            print("No faces are found in this image!!!")
            batch_of_landmarks.append([])
            num_chin = None
            num_eye = None
            return batch_of_face_locations,batch_of_landmarks,num_chin,num_eye
        
    return batch_of_face_locations,batch_of_landmarks,num_chin,num_eye
    
def draw_location_landmark(images,draw_num=1,draw_location=True,batch_of_face_locations=None,draw_landmarks=True,batch_of_landmarks=None):
    """
      可视化检测框和关键点
      
      params:
      images:图像列表
      draw_num:限制处理的图像数
      draw_location:Boolean，是否画检测框
      draw_landmarks:Boolean，是否画关键点
      batch_of_face_locations:face_recognition得到的检测框位置的列表
      batch_of_landmarks:face_recognition得到的人脸关键点位置的列表
    """
    for idx,image in enumerate(images):
        if idx == draw_num:
            break
        
        # 画矩形框
        if draw_location:
            face_locations = batch_of_face_locations[idx]        
            for face_location in face_locations:
                # Print the location of each face in this frame
                top, right, bottom, left = face_location
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        # 画关键点
        if draw_landmarks:
            landmarks = batch_of_landmarks[idx]  #  dictionary or 29*2 array
            try:
                for landmark in landmarks:
                    for _,points_list in landmark.items():
                        for point in points_list:
                            cv2.circle(image, point, radius=4, color=(0, 255, 0), thickness=-1)
            except:
                for landmark in landmarks:
                    cv2.circle(image, tuple(landmark), radius=4, color=(0, 255, 0), thickness=-1)
                        
        cv2.namedWindow('Face Recognition Display {}'.format(idx))
        cv2.imshow('Face Recognition Display {}'.format(idx), image[...,::-1])
        
    if cv2.waitKey(0):
        cv2.destroyAllWindows()
        
def landmarks_filter(batch_of_landmarks,span=1):
    """
    提取脸颊，左右眼的关键点，并做高斯滤波，消除抖动
    
    params:
    batch_of_landmarks:dlib算法提取出的人脸关键点，[[{}],[{}],...]
    span:滤波的跨度，越小越平滑
    
    return:
    batch_of_filtered_landmarks:shape:(batch, 29, 2)
    """
    # 提取感兴趣的关键点
    batch_of_extracted_landmarks = []
    for landmarks in batch_of_landmarks:
        flatten_array = []
        for landmark in landmarks:
            for k,v in landmark.items():
                if k in ['chin','left_eye','right_eye']:
                    flatten_array.append(v)
        batch_of_extracted_landmarks.append(np.concatenate(flatten_array,axis=0))
    batch_of_extracted_landmarks = np.array(batch_of_extracted_landmarks).reshape(len(batch_of_landmarks),-1)  #shape:batch*58
    
    # 高斯滤波
    df_landmarks = pd.DataFrame(batch_of_extracted_landmarks)
    df_landmarks = df_landmarks.ewm(span,adjust=False).mean()
#     print("Filtered results:")
#     print(df_landmarks.head())
    
    batch_of_filtered_landmarks = np.rint(df_landmarks.to_numpy()).astype(np.int32).reshape(-1,29,2)
    return batch_of_filtered_landmarks
    
def rotate_face(frames, batch_of_filtered_landmarks, num_chin, num_eye):
    """ 
    以两眼中心旋转帧
    params:
    frames:包含人脸的帧
    batch_of_filtered_landmarks:经过滤波之后的关键点
    
    return:
    rotated_frames:旋转后的帧图像，与原图大小一致
    eye_centers:每帧中两眼连线中心点
    angles:每帧旋转的角度
    """
    # 转化为Tensor，加速运算
    batch_of_filtered_landmarks = tf.convert_to_tensor(batch_of_filtered_landmarks,dtype=tf.float32)
    # 提取左右眼数组,shape:batch,6,2
    left_eyes = batch_of_filtered_landmarks[:,num_chin:(num_chin+num_eye),:]
    right_eyes = batch_of_filtered_landmarks[:,(num_chin+num_eye):,:]
    # 计算左右眼中心,shape:batch,2
    left_eye_centers = K.mean(left_eyes, axis=1)
    right_eye_centers = K.mean(right_eyes, axis=1)
    # 计算角度angle:shape=batch,1
    dxy = right_eye_centers - left_eye_centers
    angles = tf.atan2(dxy[...,1], dxy[...,0]) * 180. / np.pi
    # 计算眼睛连线中点,shape:batch,2
    eye_centers = tf.add(left_eye_centers,right_eye_centers) / 2.
    
    angles = angles.numpy()
    eye_centers = eye_centers.numpy()
    
    # 以眼睛连线中点为旋转中心逆时针旋转图片angle角度
    rotated_frames = []
    for frame,eye_center,angle in zip(frames,eye_centers,angles):
        rotate_matrix = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale=1)
        rotated_frame = cv2.warpAffine(frame, rotate_matrix, (frame.shape[1], frame.shape[0]))
        
        rotated_frames.append(rotated_frame)
    rotated_frames = np.array(rotated_frames)
    
    return rotated_frames, eye_centers, angles

def rotate_landmarks(eye_centers, batch_of_filtered_landmarks, angles):
    """ 
    旋转关键点
    params:
    eye_centers:每帧中两眼连线中心点
    batch_of_filtered_landmarks:需要旋转的关键点坐标
    angles:每帧旋转的角度
    
    return:
    rotated_landmarks:旋转后的关键点坐标
    """
    batch_of_filtered_landmarks = tf.convert_to_tensor(batch_of_filtered_landmarks,dtype=tf.float32)
    eye_centers = tf.convert_to_tensor(eye_centers.reshape(len(batch_of_filtered_landmarks),1,-1),dtype=tf.float32)
    angles = tf.convert_to_tensor(angles.reshape(len(batch_of_filtered_landmarks),-1) * np.pi / 180.,dtype=tf.float32)
    
    batch_of_filtered_landmarks_conv = batch_of_filtered_landmarks - eye_centers  #shape:batch,29,2
    
    rotate_mat = tf.concat([tf.cos(angles),tf.sin(angles),-tf.sin(angles),tf.cos(angles)],axis=-1)  #shape:batch,4
    rotate_mat = tf.reshape(rotate_mat,shape=(-1,2,2))  #shape:batch,2,2
    
    rotated_landmarks = tf.transpose(tf.matmul(rotate_mat,tf.transpose(batch_of_filtered_landmarks_conv,perm=[0,2,1])),perm=[0,2,1])\
    + eye_centers  #shape:batch,29,2
    
    return rotated_landmarks.numpy()

def crop_face(frames, eye_centers, rotated_landmarks):
    """ 
    根据关键点切割出人脸
    params:
    image_array: numpy array of a single image
    size: single int value, size for w and h after crop
    landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    
    return:
    cropped_img: numpy array of cropped image
    left, top: left and top coordinates of cropping
    """
    lefts = np.rint(np.min(rotated_landmarks[...,0],axis=1))  #shape:batch,
    rights = np.rint(np.max(rotated_landmarks[...,0],axis=1))  #shape:batch,
    
    widths = rights - lefts  #shape:batch,
    width = np.rint(np.mean(widths))  #scalar
    rights = lefts + width  #shape:batch,
    
    heights = np.max(rotated_landmarks[...,1],axis=1) - eye_centers[...,1]  #shape:batch,
    height = np.rint(np.mean(heights))  #scalar,这个height并不是最终切割人脸的高度，而是两眼中心到脸颊底部的距离,实际高度乘以1.2倍
    
    bottoms = np.max(rotated_landmarks[...,1],axis=1)  #shape:batch,
    bottoms = np.rint(bottoms)
    tops = np.rint(bottoms - height * 1.2)
    
    cropped_frames = []
    for left, top, right, bottom, frame in zip(lefts, tops, rights, bottoms, frames):
        pil_frame = Image.fromarray(frame)
        
        cropped_frame = pil_frame.crop((left, top, right, bottom))
        cropped_frames.append(np.array(cropped_frame))
    cropped_frames = np.array(cropped_frames)  
        
    return cropped_frames,lefts,tops

def transfer_landmark(landmarks, lefts, tops):
    """
    需要重新计算关键点的位置以匹配剪裁后的图像
    params:
    landmarks:需要变换坐标的关键点
    lefts:每帧左侧剪裁的距离
    tops:每帧上侧剪裁的距离
    
    return: 
    transferred_landmarks:转换后的关键点坐标
    """
    batch_size = len(landmarks)
    shift = np.concatenate([lefts.reshape(batch_size,1,-1),tops.reshape(batch_size,1,-1)],axis=-1)  #shape:batch,1,2
    transferred_landmarks = landmarks - shift
    
    return transferred_landmarks
    
def extract_Skin_YCrCb_Otsu(frames):
    skins = []
    masks = []
    for frame in frames:
        
        # 转至YCrCb颜色空间
        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
        cr = ycrcb_frame[:, :, 1]

        # 通过OTSU算法从Cr通道提取脸部前景区域
        _, mask = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 腐蚀膨胀去除噪声
        kernel_size = min(frame.shape[0], frame.shape[1]) // 40
        #一个ndarray，一个类似卷积核的东西，用于对图像做腐蚀和膨胀操作
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask = cv2.erode(mask, element)
        mask = cv2.dilate(mask, element)

        # 保留最大轮廓
        # contours为一个点集列表，每个元素为一个ndarray，例如：(204, 1, 2)，代表204个点组成的边界线
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        max_index = 0
        max_val = -1
        for idx, c in enumerate(contours):
            if c.shape[0] > max_val:
                max_val = c.shape[0]
                max_index = idx
        canvas = mask * 0
        mask = cv2.drawContours(canvas, contours, max_index, 1, -1)
        
        mask = np.expand_dims(mask, axis=2)
        skins.append(mask * frame)
        masks.append(mask)
        
    return np.array(skins),np.array(masks)
    
def RGB2G(frames,channel=1):
    g_channel = frames[...,1:2]
    r_channel = np.zeros(shape=g_channel.shape,dtype=frames.dtype)
    b_channel = np.zeros(shape=g_channel.shape,dtype=frames.dtype)
    
    if channel == 3:
        return np.concatenate([r_channel,g_channel,b_channel],axis=-1)  # shape:batch*w*h*3
    else:
        return frames[...,1:2]  # shape:batch*w*h*1