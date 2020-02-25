import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt 
import random 

#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 as cv
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
iterator = 0
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
args = parser.parse_args()

output_csv = open('output.csv', 'w')
sys.stdout = output_csv

iterator = 0
black_pixels = []
backSub = cv.createBackgroundSubtractorMOG2()
detection_scores = [] 
number_of_detected_objects = 0


cap = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if frame is None:
        break
        
    fgMask = backSub.apply(frame)
    #cv.imshow('difference',fgMask)
    
    cv.imwrite('difference.png',fgMask)

    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    if iterator == 0: 
        total_frames = cap.get(7)
        cap.set(1, 0)
        ret, img0 = cap.read()
        #img0 = capture.set(cv.CAP_PROP_POS_FRAMES, 0)


    height, width = fgMask.shape
    black_pixels.append((height*width) - cv.countNonZero(fgMask))
   
    iterator = iterator + 1 
    keyboard = cv.waitKey(1000)
    if keyboard == 'q' or keyboard == 27:
        break


stationary_frame_id = black_pixels.index(max(black_pixels[10:]))

cap.set(1, stationary_frame_id)
ret, imgst = cap.read()

cv.imshow('stationary_frame',imgst)
cv.imwrite('stationary_frame.png',imgst)


# Read the graph.
with tf.gfile.FastGFile('frozen_inference_graph_new.pb', 'rb') as f:
    #print('Hello')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Read and preprocess an image.
    img = cv.imread('stationary_frame.png')
    rows = img.shape[0]
    cols = img.shape[1]
    inp = cv.resize(img, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    #print(out[0][0]) #num of detections 
    #print(out[1][0]) #detection scores 
    #print(out[2][0][6][3])#detection boxes coordinates
    #print(out[3][0]) #detection class
    

   
    plt.figure(figsize=(40,40))
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    for iterator in range(0,100):
        #if output_dict['detection_scores'][iterator]*100 > 50:
        if out[1][0][iterator]*100 > 50:
            number_of_detected_objects += 1  
            detection_point = [(out[2][0][iterator][1]*1936 + 
		               out[2][0][iterator][3]*1936)/2,
		               (out[2][0][iterator][0]*1216 + 
		               out[2][0][iterator][2]*1216)/2]
            #print(iterator)
            for plotter in range(number_of_detected_objects):
                if out[3][0][iterator] == 1:
                    peanut = plt.scatter([int(detection_point[0])],[int(detection_point[1])],s=70,c='r') 
                    print(stationary_frame_id,detection_point,'Peanut')
                    plt.text(int(detection_point[0]),int(detection_point[1]),'Peanut',fontsize=30,color='r')

                elif out[3][0][iterator] == 2:
                    walnut = plt.scatter([int(detection_point[0])],[int(detection_point[1])],s=70,c='g') 
                    print(stationary_frame_id,detection_point,'Walnut')
                    plt.text(int(detection_point[0]),int(detection_point[1]),'Walnut',fontsize=30,color='g')

                elif out[3][0][iterator] == 3:
                    haselnut = plt.scatter([int(detection_point[0])],[int(detection_point[1])],s=70,c='b') 
                    print(stationary_frame_id,detection_point,'Haselnut')
                    plt.text(int(detection_point[0]),int(detection_point[1]),'Haselnut',fontsize=30,color='b')
                
            number_of_detected_objects = 0
           
            
plt.show()




#peanut1 = plt.scatter([190],[20],s=1,c='r',label='Peanut1')
            #walnut1 = plt.scatter([190],[20],s=1,c='g',label='Walnut1')
            #haselnut1 = plt.scatter([190],[20],s=1,c='b',label='Haselnut1')
            #plt.legend((peanut,walnut,haselnut),('Peanut','Walnut','Haselnut'),scatterpoints=1)
            #plt.legend(['Peanut','Walnut','Haselnut'])




