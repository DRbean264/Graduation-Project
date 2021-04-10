import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import autoreload
from Frames_Alignment import *

def draw_ref_grid(frame,w_density=5,h_density=5):
    """
    在人脸上绘制网格
    frame:shape:h*w*n_channel
    """
    frame = np.squeeze(frame)
    H,W = frame.shape[:2]
    h,w = np.ceil(np.array(frame.shape[:2])/np.array((h_density,w_density))).astype(np.int32)
    
    # 画竖条纹
    for i in range(w_density):
        pt1 = (min((i + 1) * w,W) - 2,0)
        pt2 = (min((i + 1) * w,W) - 1,H-1)
        cv2.rectangle(frame,pt1,pt2,color=(255,255,255),thickness=-1)
    # 画横条纹
    for i in range(h_density):
        pt1 = (0,min((i + 1) * h,H) - 2)
        pt2 = (W - 1,min((i + 1) * h,H) - 1)
        cv2.rectangle(frame,pt1,pt2,color=(255,255,255),thickness=-1)
    
    cv2.namedWindow("Image with grid")
    cv2.imshow("Image with grid",frame[...,::-1])

    if cv2.waitKey(0):
        cv2.destroyAllWindows()
        
def channel_1D_signal(frames,w_density,h_density,ROI_ids):
    """
    提取指定ROI的一维时间信号
    
    params:
    frames:shape:batch*H*W*n_channel
    w_density:横向分割的密度
    h_density:纵向分割的密度
    ROI_ids:list,ROI的序号
    """
#     frames = np.squeeze(frames)
    H,W = frames.shape[1:3]
    h,w = np.ceil(np.array((H,W))/np.array((h_density,w_density))).astype(np.int32)
    
    signal_1D = []
    for ROI_id in ROI_ids:
        x_id,y_id = ROI_id // w_density,ROI_id % w_density
        
        ROI = frames[:,x_id * h:(x_id + 1) * h,y_id * w:(y_id + 1) * w,:]  # shape:batch*h*w*n_channel
        signal = np.mean(ROI,axis=(1,2))  # shape:batch*n_channel
        signal_1D.append(signal.reshape(signal.shape[0],1,-1))  # shape:batch*1*n_channel
    
    return np.concatenate(signal_1D,axis=1)  # shape:batch*n_ROI*n_channel
    
def display_1D_signal(signal_1D,ROI_ids,channel_id,save=False,save_path=None):
    """
    显示一维的信号
    
    params:
    signal_1D:提取出的1D信号,shape:batch*n_ROI*n_channel
    ROI_ids:list,ROI的序号
    channel_id:要显示的通道id，指标从0开始
    save:要不要保存
    save_path:保存路径
    """
    plt.figure(figsize=(13, 10))
    #  调整子图间距
    plt.subplots_adjust(wspace = 0, hspace = 0.5)
    for idx,ROI_id in enumerate(ROI_ids):
        data = signal_1D[:,idx,channel_id]
        
        ax = plt.subplot(len(ROI_ids),1,idx + 1)
        
#         ax.set_xlabel("ROI {} (row:{} column:{}) signal".format(ROI_id, ROI_id // 4 + 1,ROI_id % 4 + 1))
        ax.set_ylabel("ROI {}".format(ROI_id),fontsize = 15)
        ax.set_title("Channel {} 1D signal".format(channel_id))
        
        xticks = np.arange(1,len(data)+1)
        ax.set_xticks(xticks)
        ax.set_xlim([1,len(data)+1])
        
        yticks = np.arange(min(data)-5,max(data)+5,(max(data)-min(data)+10)/5)
        ax.set_yticks(yticks)
        ax.set_ylim([min(data)-5,max(data)+5])

        x = np.arange(1,len(data)+1,1)
        ax.plot(x,data,"-")
        ax.grid(True)
    plt.show()
    
    if save:
        with open(save_path,'a') as f:
            pass
            
def save_signal(save_root_path,signal_1D,ROI_ids,cmap):
    """
    以追加的形式保存1D信号到txt文件中
    
    params:
    save_paths:保存的路径
    signal_1D:带保存的1D信号,shape:batch_size*n_ROI*n_channel
    ROI_id:list,ROI的id
    cmap:指定了是从哪一种颜色通道提取的信号，eg.'Green','YUV','RGB'
    """
    if cmap.lower() == 'green':
        channel_name = ['G']
    elif cmap.lower() == 'yuv':
        channel_name = ['Y','U','V']
    elif cmap.lower() == 'rgb':
        channel_name = ['R','G','B']
    
    for i in range(signal_1D.shape[2]):  # i是channel的索引
        for j in range(signal_1D.shape[1]):  # j是ROI的索引
            save_data = signal_1D[:,j,i]  #  shape:batch_size,
            save_path = os.path.join(save_root_path,'channel_{}_ROI_{}.txt'.format(channel_name[i],ROI_ids[j]))
            
            with open(save_path,'a') as f:
                save_data = [str(d) for d in save_data]
                f.write(' '.join(save_data) + ' ')

def show_signal(signal_paths,num_ROI,fps=30,save_img=False,save_path=None):
    """
    从文件读取保存下来的1D信号并可视化
    
    params:
    signal_paths:保存有1D信号的文件路径列表
    num_ROI:选取了多少个ROI
    save_img:是否将结果图保存到本地
    save_path:保存的路径
    """
    #画图
    plt.figure(figsize=(10, 10))
    num_row = num_ROI
    num_col = int(np.ceil(len(signal_paths) // num_row))
#     print(num_row,num_col)
    
    ROI_ids = []
    channels = []
    for signal_path in signal_paths:
        filename = os.path.basename(signal_path)
        
        ROI_id = int(filename.split("_")[-1][:-4])  #  最后的-4是为了把.txt的后缀去掉
        channel = filename.split("_")[1]
        if ROI_id not in ROI_ids:
            ROI_ids.append(ROI_id)
        if channel not in channels:
            channels.append(channel)
        
        with open(signal_path,'r') as f:
            data = [float(l) for l in f.read().split()]
        
        #  绘制的位置
        draw_loc = ROI_ids.index(ROI_id) * num_col + channels.index(channel) + 1
        ax = plt.subplot(num_row,num_col,draw_loc)
        
        #  设置标签和标题
        ax.set_xlabel("Time/s")
        ax.set_ylabel("ROI {}".format(ROI_id))
        if channel in 'RGB':
            ax.set_title("{} channel of RGB".format(channel))
        elif channel in 'YUV':
            ax.set_title("{} channel of YUV".format(channel))
        
        #  设置坐标轴刻度及范围
        xticks = np.round(np.arange(0,len(data)/fps+0.001,len(data)/fps/10),2)
#         print(xticks)
        ax.set_xticks(xticks)
        ax.set_xlim([0,len(data)/fps])
        #ax.set_xlim([40,53])
        
        yticks = np.arange(min(data)-5,max(data)+5,(max(data)-min(data)+10)/10)
        ax.set_yticks(yticks)
        ax.set_ylim([min(data)-5,max(data)+5])
        #ax.set_ylim([95,130])

        x = np.round(np.arange(0,len(data),1) / fps,2)
        ax.plot(x,data,"-")
        ax.grid(True)
    
    if save_img:
        plt.savefig(save_path,dpi=300)
    
    plt.show()