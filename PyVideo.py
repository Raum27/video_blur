import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2,mmcv
from PIL import Image, ImageDraw
from keras_facenet import FaceNet
import numpy as np
import time
from numpy import dot
from numpy.linalg import norm

from numba import jit,cuda, float64, int64
import numba


import warnings
warnings.filterwarnings('ignore')

import yolov5
model =  yolov5.load(r"C:\Users\Raum\Desktop\crowdhuman_yolov5m.pt").to('cuda')

np.seterr(divide='ignore', invalid='ignore')
embedder = FaceNet()

def cosine_similarity(a, b):
    a = np.array(a,dtype=np.float32)
    b = np.array(b,dtype=np.float32)
    value_AB = dot(a, b) / (norm(a) * norm(b))
    return 0.0 if value_AB != value_AB else value_AB

def clear_Faces_lock(folder):
  if os.listdir(folder+'/') !=[]:
    for i in os.listdir(folder+'/'):
        folder_folder = os.path.join(folder+'/',i)
        for i in os.listdir(folder_folder+'/'):
            os.remove(os.path.join(folder_folder+'/',i))
        os.rmdir(folder_folder)    
      

@jit(nopython=True)
def cosine_similarity_numba(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)

@jit(nopython=True)
def lock_blur(face_lock:numba.int64[:],embedding_one_frame:numba.float64[:,:,:],embedding_faces:numba.float64[:,:,:]):
    Detection_check = np.zeros((embedding_one_frame.shape[0]))
    if face_lock.shape[0] !=0 : 
        for index_face in face_lock:
            for j in range(len(embedding_one_frame)):
                similar = cosine_similarity_numba(embedding_one_frame[j][0],embedding_faces[index_face][0])
                if similar>=0.70 and Detection_check[j] == 0 :
                    Detection_check[j] = 1
                    break
    return Detection_check

def video_find_cosine(file):
    ''' cosine_similarity frame '''
    video = mmcv.VideoReader(file)
    print('Frames: ',len(video))
    masker = 1
    arr = np.zeros(len(video))
    walk1 =0
    walk2 =1
    for i in range(len(video)-1):
        frame_1 = cv2.resize(cv2.cvtColor(video[walk1],cv2.COLOR_BGR2RGB),(224,224)).reshape(-1)
        frame_2 = cv2.resize(cv2.cvtColor(video[walk2],cv2.COLOR_BGR2RGB),(224,224)).reshape(-1)
        if cosine_similarity(frame_1,frame_2) ==0.0:
            arr[walk1] = -1
            arr[walk2] = -1
        if cosine_similarity(frame_1,frame_2) >= 0.80:
            arr[walk1] = masker
            arr[walk2] = masker
        else:
            arr[walk1] = masker
            masker +=1
            arr[walk2] = masker
        walk1 +=1
        walk2 +=1
    return video,arr

def where_3(arr):
    values, index_frames,index_count = np.unique(arr,return_index=True,return_counts=True)
    print(values.min(),values.max())
    Frame_similarity = index_count//2 ;'''get middle frame'''
    Frame_last = (index_count+index_frames) ;'''get last frame'''

    return index_frames,Frame_similarity,Frame_last

def AI_Prediction_Frame(index_frames,Frame_similarity,Frame_last,video):
  '''
      original Prediction AI Frame
      หาว่าเฟรมใดบ้างที่มีใบหน้าอยู่ และ จะเก็บใบหน้าที่มีเพื่อให้ผู้ใช้เลือก
                                                         '''
  faces_scene = []
  bool_scene = np.zeros(len(video))
  where_crop = []
  for i in range(len(Frame_similarity)):
    Have_face = False
    start = index_frames[i]
    midframe = index_frames[i]+Frame_similarity[i]
    last = Frame_last[i]-1
    there_frame =[start,midframe,last]
    y_test = [cv2.cvtColor(video[xo],cv2.COLOR_BGR2RGB) for xo in there_frame]
    xxx = model(y_test)

    for ox in range(len(there_frame)):
        predictions = xxx.pred[ox]
        boxes = predictions[:, :4]
        categories = predictions[:, 5]

        Ar = np.where(categories.to('cpu').numpy()==1)[0]
        if Ar.shape[0] !=0:
            Have_face = True
            where_crop.append(there_frame[ox])
            faces_scene.append(boxes[Ar].to('cpu').numpy())
    if Have_face:
        bool_scene[index_frames[i]:Frame_last[i]] = 1

  return faces_scene,bool_scene,where_crop

def Face_croper(faces_boxes,where_crop,video):
    '''
        crop for selected face by cilent and find value
        ตัดเอาเฉพาะใบหน้าที่พบลงในแต่ละโฟลเดอร์ โดยภายในก็จะหน้าทั้งหมดอยู่
        เพื่อให้ผู้ใช้เลือก
                                                                '''
    folder = "Faces-lock"
    embedding_faces = []
    try:
        os.makedirs(folder)
    except:
        # print('folder already exists')
        pass
    NUMBER = 1
  
    for i,number in enumerate(where_crop):
        for j,(x1,y1,x2,y2) in enumerate(faces_boxes[i]):
                face = cv2.cvtColor(video[number][int(y1):int(y2),int(x1):int(x2)],cv2.COLOR_BGR2RGB)
                image_face = Image.fromarray(face)
                image_face = image_face.resize((224,224))

                faces_crops = np.array(image_face).reshape(-1,224,224,3)
                EMBED = embedder.embeddings(faces_crops)

                path = folder+'/Face_{}.png'.format(NUMBER) # path for collect images
                image_face.save(path)
                embedding_faces.append(EMBED)
                NUMBER+=1
    print('There are faces in Frames:',len(embedding_faces))
    return embedding_faces

def locate_face(boxes,IN_FRAME,video):
    embedding_one_frame = []
    for x1,y1,x2,y2  in boxes:

        face = video[IN_FRAME][int(y1):int(y2),int(x1):int(x2)]
        image_face = Image.fromarray(face)
        image_face = image_face.resize((224,224))

        faces_crops = np.array(image_face).reshape(-1,224,224,3)
        
        embedding_one_frame.append(embedder.embeddings(faces_crops))
    return embedding_one_frame

def Anotation_frame(Frame,Detection_check,boxes,Filter_oFF):
    for index_box,DE_CRECK in enumerate(Detection_check):
        x1, y1, x2, y2 = int(boxes[index_box][0]),int(boxes[index_box][1]),int(boxes[index_box][2]),int(boxes[index_box][3])
        if DE_CRECK !=1:
            if Filter_oFF:
                censor_region = (x1,y1,x2,y2)
                censored_area = Frame[censor_region[1]:censor_region[3], censor_region[0]:censor_region[2]]
                censored_width, censored_height = censored_area.shape[1], censored_area.shape[0]
                pixel_size = 8
                censored_area = cv2.resize(censored_area, (pixel_size,pixel_size))
                censored_area = cv2.resize(censored_area, (censored_width, censored_height), interpolation=cv2.INTER_NEAREST)
                Frame[censor_region[1]:censor_region[3], censor_region[0]:censor_region[2]] = censored_area
                
            else:
                fitter_ = Image.open(r"C:\Users\Raum\Desktop\jec\code\dataface\memeface.png").convert("RGBA")
                x = int(x2-x1) 
                y = int(y2-y1) 
                fitter_ = fitter_.resize((x,y))
                fill_image = Image.fromarray(cv2.cvtColor(Frame,cv2.COLOR_BGR2RGB))
                fill_image.paste(fitter_,(x1,y1),fitter_)
                Frame[:,:,::-1] = fill_image
    return Frame

def create_anotation_frame(video,bool_scene,embedding_faces,face_lock,Filter_oFF=True):
    folder_anotation = "Anotation_frames"
    try:
        os.makedirs(folder_anotation)
    except:
        # print('folder already exists')
        pass

    for Frame in range(len(video)): 
        if bool_scene[Frame] == 1: 
            image = video[Frame]
            Y = model(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            predictions = Y.pred[0].to('cpu')
            boxes = predictions[:, :4]
            categories = predictions[:, 5]
            Ar = np.where(categories.numpy()==1)[0]
            zero_padded_string = str(Frame).zfill(6)
            if Ar.shape[0] !=0 :
                boxes = boxes[Ar].numpy()
                embedding_one_frame = locate_face(boxes,Frame,video)
                Detection_check = lock_blur(face_lock,np.array(embedding_one_frame),np.array(embedding_faces))
                Frame_info = Anotation_frame(image,Detection_check,boxes,Filter_oFF)
                cv2.imwrite(f'{folder_anotation}/{zero_padded_string}.jpg',Frame_info)
            else:
                cv2.imwrite(f'{folder_anotation}/{zero_padded_string}.jpg',image)
         
        else:
            zero_padded_string = str(Frame).zfill(6)
            cv2.imwrite(f'{folder_anotation}/{zero_padded_string}.jpg',video[Frame])

def write_video_file(folder_anotation,video):
    ''' write frames to video'''
    dim = video[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'FMP4') ;'''MP4V, h264, x264'''
    Video_Writer = cv2.VideoWriter('video_New.mp4', fourcc, 25.0, (dim[1],dim[0]))

    for frame in os.listdir(folder_anotation):
        file = cv2.imread(f'{folder_anotation}/{str(frame)}')
        Video_Writer.write(file)
        
    Video_Writer.release()

if __name__ == "__main__":
    start = time.time()
    video,arr = video_find_cosine(r"C:\Users\Raum\Desktop\Video_blur\testvideo.mp4") ;'''step 1'''

    index_frames,Frame_similarity,Frame_last = where_3(arr)
    faces_scene,bool_scene,where_crop = AI_Prediction_Frame(index_frames,Frame_similarity,Frame_last,video)
    embedding_faces = Face_croper(faces_scene,where_crop,video)

    face_lock = np.array([4,16,25])   ;'''step 2'''
    
    # start = time.time()
    create_anotation_frame(video,bool_scene,embedding_faces,face_lock,Filter_oFF=True)
    write_video_file('Anotation_frames',video)
    end  = time.time()
    print(end-start)