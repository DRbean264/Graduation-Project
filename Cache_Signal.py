import os
import numpy as np
import cv2
from scipy import signal
import sys
from Frames_Alignment import *

# 系统默认的配置
batch_size = 10
cmap = 'RGB'
resize = False
remove_skin = True
min_size = None
span = 10
w_density = 7
h_density = 4
ROI_ids = list(range(w_density*h_density))

def get_video_length(video_path):
    """
        给入视频路径，计算视频的长度以及FPS

        return:
        video_length:视频的长度，单位：s
        fps_counter：视频的帧率
    """
    video_capture = cv2.VideoCapture(video_path)

    frame_counter = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_counter = int(video_capture.get(cv2.CAP_PROP_FPS))
    video_length = frame_counter/fps_counter

    video_capture.release()
    
    return video_length,fps_counter

def size_convert(size):
    """
       文件大小单位换算,输入size为文件字节数
    """
    K, M, G = 1024, 1024 ** 2, 1024 ** 3
    if size >= G:
        return str(round((size / G), 2)) + ' GB'
    elif size >= M:
        return str(round((size / M), 2)) + ' MB'
    elif size >= K:
        return str(round((size / K), 2)) + ' K'
    else:
        return str(round((size), 3)) + ' B'

def get_video_size(video_path):
    file_byte = os.path.getsize(video_path)
    
    return size_convert(file_byte)

def video_crop_info(video_length,fps,crop_length):
    """
        计算视频分割的信息，给出所选片段的开头帧和结尾帧
        注意因为训练集采用250帧训练，因此测试时也是250，不能更改

        params:
        video_length:视频的长度，单位：s
        fps：视频的帧速率，单位：帧每秒
        crop_length:每个分割片段的长度，单位：s

        return:
        start_frame:开始帧位置的列表[start1,start2,...]
        stop_frame:结束帧位置的列表[stop1,stop2,...]
    """
    num = int(video_length // crop_length)   #  片段数目
    start_frame = [int(np.round((video_length - crop_length * num) * fps / 2 + i * crop_length * fps)) for i in range(num)]
    stop_frame = (np.array(start_frame) + 250).astype(np.int32).tolist()
    
    return start_frame,stop_frame

def yield_frames_of_clipped_video(video_file,batch_size,min_size,start,stop,resize=False):
    """
        从视频文件读取一定数量的帧，存放在一个列表中，元素为np.ndarray

        params:
        video_file:视频文件路径列表
        batch_size:一批的数量
        min_size:读入的图像最短边缩放到多长
        start:clip开始的帧
        stop:clip结束的帧
    """
    video_capture = cv2.VideoCapture(video_file)
    frame_idx = 0
    batch_idx = 0

    while True:
        frames = []
        while video_capture.isOpened():
            while batch_idx < batch_size:
                ret, frame = video_capture.read()
                #  读到视频结尾的话，退出循环
                if not ret:
                    video_capture.release()
                    break
                if frame_idx >= start:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if resize == True:
                        scaled_shape = get_scaled_wh(frame.shape[:2],min_size)
                        frame = cv2.resize(frame,scaled_shape[::-1],interpolation=cv2.INTER_LINEAR)
                    frames.append(frame)

                    batch_idx += 1
                frame_idx += 1
                if frame_idx == stop:
                    video_capture.release()
                    break 
            batch_idx = 0
            break
        yield frames

def video2ROI(file,batch_size,cmap,resize,remove_skin,min_size,span,w_density,h_density,ROI_ids,save_root_path,start,stop):
    """
        从视频文件中提取1D信号并保存在文件中的整个pipeline

        params:
        file:视频文件路径
        batch_size:批量处理的帧数目，此电脑显存不大，所以别超过20
        cmap:颜色通道
        resize:是否缩放图片
        min_size:把图片缩放到最短边的边长为min_size
        span:高斯滤波消抖的跨度，越大越平滑
        w_density:ROI纵向分割的密度
        h_density:ROI横向分割的密度
        ROI_ids:list，要读取的ROI序号，指标从0开始
        start:clip开始的帧
        stop:clip结束的帧
    """
    # 帧生成器
    frames_gen = yield_frames_of_clipped_video(file,batch_size,min_size,start,stop,resize)
    
    while True:
        frames = next(frames_gen)
        #  如果没有帧，则直接结束
        if len(frames) == 0:
            break
        
        #  获取帧中的人脸位置及关键点坐标
        batch_of_face_locations,batch_of_landmarks,num_chin,num_eye = get_location_landmark(frames,0)
        
        #  如果在当前clip中有一帧没找到人脸，那么这个clip就不做处理
        if num_chin == None and num_eye == None:
            return False
        
        #  关键点滤波
        batch_of_filtered_landmarks = landmarks_filter(batch_of_landmarks,span)
        frames = np.array(frames)
        
        #  旋转帧使人脸水平
        rotated_frames, eye_centers, angles = rotate_face(frames, batch_of_filtered_landmarks, num_chin, num_eye)
        
        #  旋转关键点
        rotated_landmarks = rotate_landmarks(eye_centers, batch_of_filtered_landmarks, angles)
        
        #  把人脸剪裁出来
        cropped_frames,lefts,tops = crop_face(rotated_frames, eye_centers, rotated_landmarks)
        
        #  关键点坐标变换
        transferred_landmarks = transfer_landmark(rotated_landmarks, lefts, tops)
        
        #  去除非皮肤区域
        if remove_skin:
            skin_frames,skin_masks = extract_Skin_YCrCb_Otsu(cropped_frames)
        else:
            skin_frames = cropped_frames
        
        #  颜色通道转换/提取
        if cmap.lower() == 'green':
            skin_frames = RGB2G(skin_frames)
        elif cmap.lower() == 'yuv':
            skin_frames = RGB2YUV(skin_frames)
        elif cmap.lower() == 'rgb':
            pass

        signal_1D = channel_1D_signal(skin_frames,w_density,h_density,ROI_ids)  #  shape:batch_size*n_ROI*n_channel
        
        save_signal(save_root_path,signal_1D,ROI_ids,cmap)
        
        #  如果frames长度不足batch_size，说明视频读完了
        if len(frames) != batch_size:
            break
        
    return True

def cache_signal_VIPL(video_path,cache_root_dir,crop_length=10):
    """
       遍历视频，缓存视频片段1D信号
       
       params:
       video_path:视频绝对路径列表
       cache_root_dir:缓存目录
       crop_length:视频片段截取的长度,如果是None，则整个视频都要
    """
    for path in video_path:
        person_id,scene_id = path.split("\\")[-4],path.split("\\")[-3]
        person_root_dir = os.path.join(cache_root_dir,person_id)
        scene_root_dir = os.path.join(person_root_dir,scene_id)
        #  创建目录
        if not os.path.exists(person_root_dir):
            os.mkdir(person_root_dir)
        if not os.path.exists(scene_root_dir):
            os.mkdir(scene_root_dir)
        
        video_length,fps = get_video_length(path)
        start_frame,stop_frame = video_crop_info(video_length,fps,crop_length)
        
        for idx,start,stop in zip(range(1,len(start_frame)+1),start_frame,stop_frame):
            if not os.path.exists(scene_root_dir):
                os.mkdir(scene_root_dir)
            
            save_root_path = os.path.join(scene_root_dir,"clip{}".format(idx))
            if not os.path.exists(save_root_path):
                os.mkdir(save_root_path)
            
            clip_valid = video2ROI(path,batch_size,cmap,resize,remove_skin,min_size,span,w_density,h_density,ROI_ids,save_root_path,start,stop)
            
            #  如果clip无效，即出现未检测出人脸的帧，则需要把当前clip文件夹删除
            if not clip_valid:
                for dirpath, dirnames, filenames in os.walk(save_root_path):
                    for file in filenames:
                        file_abspath = os.path.join(dirpath,file)
                        os.remove(file_abspath)
                    os.removedirs(dirpath)
                print("Clip {} of video {} is not considered because of non-face-detected frames.".format(idx,path))
                continue
            
        print("Video {} cached successfully!".format(path))

def cache_signal_ECNU(video_path,cache_root_dir,crop_length=10):
    """
       遍历视频，缓存视频片段1D信号
       
       params:
       video_path:视频绝对路径列表
       cache_root_dir:缓存目录
       crop_length:视频片段截取的长度,如果是None，则整个视频都要
    """
    for path in video_path:
        person_id = os.path.basename(path).split('.')[0]
        person_root_dir = os.path.join(cache_root_dir,person_id)
        #  创建目录
        if not os.path.exists(person_root_dir):
            os.mkdir(person_root_dir)
        
        video_length,fps = get_video_length(path)
        start_frame,stop_frame = video_crop_info(video_length,fps,crop_length)
        
        for idx,start,stop in zip(range(1,len(start_frame)+1),start_frame,stop_frame):
            if not os.path.exists(person_root_dir):
                os.mkdir(person_root_dir)
            
            save_root_path = os.path.join(person_root_dir,"clip{}".format(idx))
            if not os.path.exists(save_root_path):
                os.mkdir(save_root_path)
            
            clip_valid = video2ROI(path,batch_size,cmap,resize,remove_skin,min_size,span,w_density,h_density,ROI_ids,save_root_path,start,stop)
            
            #  如果clip无效，即出现未检测出人脸的帧，则需要把当前clip文件夹删除
            if not clip_valid:
                for dirpath, dirnames, filenames in os.walk(save_root_path):
                    for file in filenames:
                        file_abspath = os.path.join(dirpath,file)
                        os.remove(file_abspath)
                    os.removedirs(dirpath)
                print("Clip {} of video {} is not considered because of non-face-detected frames.".format(idx,path))
                continue
            
        print("Video {} cached successfully!".format(path))

def cache_signal_infer(video_path,cache_root_dir,crop_length=10,mode='train'):
    """
       遍历视频，缓存视频片段1D信号
       
       params:
       video_path:视频绝对路径列表
       cache_root_dir:缓存目录
       crop_length:视频片段截取的长度
       mode:指定训练还是推理模式,train/inference
    """
    for path in video_path:
        scene_root_dir = os.path.join(os.path.join(os.path.dirname(path),"Cached_Signal"),os.path.basename(path).split(".")[0])
        
        #  创建目录    
        if not os.path.exists(scene_root_dir):
            os.mkdir(scene_root_dir)  
                
        video_length,fps = get_video_length(path)
        if video_length < crop_length:
            print("Video: {} is too short. This video will not be processed.".format(path))
            continue
        else:
            start_frame,stop_frame = video_crop_info(video_length,fps,crop_length)
        
        for idx,start,stop in zip(range(1,len(start_frame)+1),start_frame,stop_frame):
            save_root_path = os.path.join(scene_root_dir,"clip{}".format(idx))
            if not os.path.exists(save_root_path):
                os.mkdir(save_root_path)
            
            clip_valid = video2ROI(path,batch_size,cmap,resize,remove_skin,min_size,span,w_density,h_density,ROI_ids,save_root_path,start,stop)
            
            #  如果clip无效，即出现未检测出人脸的帧，则需要把当前clip文件夹删除
            if not clip_valid:
                for dirpath, dirnames, filenames in os.walk(save_root_path):
                    for file in filenames:
                        file_abspath = os.path.join(dirpath,file)
                        os.remove(file_abspath)
                    os.removedirs(dirpath)
                print("Clip {} of video {} is not considered because of non-face-detected frames.".format(idx,path))
                continue
        
        print("Video {} cached successfully!".format(path))

def filter_cached_signal_VIPL(cache_root_dir,filtered_root_dir,fps,scene='v1',mode='train'):
    """
       把缓存好的文件中的数据进行带通滤波,并缓存到新的路径下
       
       params:
       cache_root_dir：缓存好的1D信号的根目录
       filtered_root_dir：保存滤波后信号的根目录
       scene：场景选择
       mode：指定训练还是推理模式,train/inference
    """
    
    for dirpath, dirnames, filenames in os.walk(cache_root_dir):
        if mode == "train":
            if dirpath.split("\\")[-2] == scene:
                for file in filenames:
                    signal_file = os.path.join(dirpath, file)
                    
                    save_file_path = os.path.join(filtered_root_dir,signal_file[signal_file.find('\\p') + 1:])
                    if not os.path.exists(os.path.dirname(save_file_path)):
                        os.makedirs(os.path.dirname(save_file_path))
                    elif os.path.exists(save_file_path):
                        continue
                    
                    with open(signal_file,'r') as f:
                        data = [float(l) for l in f.read().split()]

                    #  [0.67,4]/25(fps) = [0.027,0.16]
                    frequency_range = (np.array([0.67,4]) / fps).tolist()
                    b_band, a_band = signal.butter(8, frequency_range, 'bandpass')   #配置滤波器, 8表示滤波器的阶数
                    filtered_data = signal.filtfilt(b_band, a_band, data, axis=-1)  #data为要过滤的信号

                    with open(save_file_path,'w') as f:
                        save_data = [str(d) for d in filtered_data]
                        f.write(' '.join(save_data))
        elif mode == "inference":
            for file in filenames:
                signal_file = os.path.join(dirpath, file)

                save_file_path = os.path.join(filtered_root_dir,"\\".join(signal_file.split("\\")[-3:]))
                if not os.path.exists(os.path.dirname(save_file_path)):
                    os.makedirs(os.path.dirname(save_file_path))
                elif os.path.exists(save_file_path):
                    continue
                
                with open(signal_file,'r') as f:
                    data = [float(l) for l in f.read().split()]

                #  [0.67,4]/25(fps) = [0.027,0.16]
                frequency_range = (np.array([0.67,4]) / fps).tolist()
                b_band, a_band = signal.butter(8, frequency_range, 'bandpass')   #配置滤波器, 8表示滤波器的阶数
                filtered_data = signal.filtfilt(b_band, a_band, data, axis=-1)  #data为要过滤的信号

                with open(save_file_path,'w') as f:
                    save_data = [str(d) for d in filtered_data]
                    f.write(' '.join(save_data))
                        
def filter_cached_signal_ECNU(cache_root_dir,filtered_root_dir,fps):
    """
       把缓存好的文件中的数据进行带通滤波,并缓存到新的路径下
       cache_root_dir：缓存好的1D信号的根目录
       filtered_root_dir：保存滤波后信号的根目录
    """
    for dirpath, dirnames, filenames in os.walk(cache_root_dir):
        for file in filenames:
            signal_file = os.path.join(dirpath, file)
            
            save_file_path = os.path.join(filtered_root_dir,"\\".join(signal_file.split("\\")[-3:]))
            if not os.path.exists(os.path.dirname(save_file_path)):
                os.makedirs(os.path.dirname(save_file_path))
            elif os.path.exists(save_file_path):
                continue
            
            with open(signal_file,'r') as f:
                data = [float(l) for l in f.read().split()]

            #  [0.67,4] * 2 / 25(fps) = [0.054,0.32]
            frequency_range = (np.array([0.67,4]) * 2 / fps).tolist()
            b_band, a_band = signal.butter(8, frequency_range, 'bandpass')   #配置滤波器, 8表示滤波器的阶数
            filtered_data = signal.filtfilt(b_band, a_band, data, axis=-1)   #data为要过滤的信号
            
            with open(save_file_path,'w') as f:
                save_data = [str(d) for d in filtered_data]
                f.write(' '.join(save_data))
          
def filter_cached_signal_infer(cache_root_dir,filtered_root_dir,fps,scene='v1'):
    """
       把缓存好的文件中的数据进行带通滤波,并缓存到新的路径下
       cache_root_dir：缓存好的1D信号的根目录
       filtered_root_dir：保存滤波后信号的根目录
       scene：场景选择
    """
    
    for dirpath, dirnames, filenames in os.walk(cache_root_dir):
        for file in filenames:
            signal_file = os.path.join(dirpath, file)
                
            save_file_path = os.path.join(filtered_root_dir,"\\".join(signal_file.split("\\")[-3:]))
            if not os.path.exists(os.path.dirname(save_file_path)):
                os.makedirs(os.path.dirname(save_file_path))
            elif os.path.exists(save_file_path):
                continue
            
            with open(signal_file,'r') as f:
                data = [float(l) for l in f.read().split()]

            #  [0.67,4]/25(fps) = [0.027,0.16]
            frequency_range = (np.array([0.67,4]) * 2 / fps).tolist()
            b_band, a_band = signal.butter(8, frequency_range, 'bandpass')   #配置滤波器, 8表示滤波器的阶数
            filtered_data = signal.filtfilt(b_band, a_band, data, axis=-1)   #data为要过滤的信号

            with open(save_file_path,'w') as f:
                save_data = [str(d) for d in filtered_data]
                f.write(' '.join(save_data))
                        
def show_signal(data,fps=25):
    plt.figure(figsize=(10, 5))
    
    plt.xlabel("Time/s")

    #  设置坐标轴刻度及范围
    xticks = np.round(np.arange(0,len(data)/fps+0.001,len(data)/fps/10),2)
    plt.xticks(xticks)
    plt.xlim([0,len(data)/fps])

    yticks = np.arange(min(data)-1,max(data+1,(max(data)-min(data+2)/10)))
    plt.yticks(yticks)
    plt.ylim([min(data)-1,max(data)+1])

    x = np.round(np.arange(0,len(data),1) / fps,2)
    plt.plot(x,data,"g-")
    plt.axis("off")
    plt.grid(True)