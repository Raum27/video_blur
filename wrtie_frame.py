import cv2
import time
import threading


def write_frame(frame,output_filename):
    cv2.imwrite(output_filename, frame)

def read_video(file_path,out_write_path):
    cap = cv2.VideoCapture(file_path)
    thread_numbers = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/4)
    print(thread_numbers)
    count_t = 0
    threads = []
    current_frame = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output_filename = f"{out_write_path}frame_{current_frame}.jpg"
        if count_t <=thread_numbers:
            current_frame +=1
            count_t +=1
            thread = threading.Thread(target=write_frame, args=(frame,output_filename,))
            threads.append(thread)
            thread.start()
            continue
        write_frame(frame,output_filename)
        current_frame +=1
        count_t +=1
        if count_t == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/2):
            count_t =0
     

    for thread in threads:
        thread.join()

    cap.release() 
    print('finish')
    cv2.destroyAllWindows()

if __name__ == "__main__":
    filename = r"C:\Users\Raum\Desktop\jec\code\dataface\netflix.mp4"
    out_write_path = r"C:\Users\Raum\Desktop\testversion2\test_image\\"

    start = time.time()
    read_video(filename,out_write_path)
    end = time.time()

    print(end-start)
