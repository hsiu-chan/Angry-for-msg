import numpy as np
from skimage import io
from os import listdir
#import cv2
from cv2 import cvtColor, threshold, morphologyEx, findContours,COLOR_BGRA2BGR,THRESH_BINARY,THRESH_OTSU, RETR_EXTERNAL, MORPH_OPEN, CHAIN_APPROX_SIMPLE, COLOR_BGR2GRAY
from PIL import Image as mg

def polygon_area(points):
    area = 0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return int(abs(area / 2))

input_path="./input"
output_path='./result'
angry=mg.open('angry.png').convert('RGBA')

for filename in listdir(input_path):
    if not (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')):
        continue
    fig_path=input_path+'/'+filename
    read=io.imread(fig_path)
    try:
        read=cvtColor(read, COLOR_BGRA2BGR)
    except:
        continue
    h, w, d = read.shape
    gray = cvtColor(read, COLOR_BGR2GRAY)
    ret,thresh = threshold(gray,0,255,THRESH_BINARY+THRESH_OTSU)#二值化
    # 尋找輪廓
    k = np.ones((int(h/100), int(w/100)), np.uint8)  #3x3捲積核
    thresh_open = morphologyEx(thresh, MORPH_OPEN, k)      # 開運算[先膨脹-後腐蝕]
    contours, hierarchy = findContours(thresh_open, RETR_EXTERNAL,CHAIN_APPROX_SIMPLE)


    emoji=[]

    for contour in contours:
        c=np.array(contour)
        if len(c)<3:
            continue
        c=np.squeeze(c)
        ca=(np.max(c[:,0])-np.min(c[:,0]))*(np.max(c[:,1])-np.min(c[:,1]))

        if polygon_area(c)/(h*w)<1/800:
            continue
        if polygon_area(c)/ca<0.85:
            continue
        emoji.append([np.max(c[:,0]), np.max(c[:,1])])
    print(emoji)
    emoji=np.array(emoji)
    
    n=int(w/20)
    angry=angry.resize((n,n))
    r,g,b,a= angry.split()

    result=mg.open(fig_path)
    for p in emoji:
        result.paste(angry, (p[0]-2*n,p[1]-int(n*0.4)),mask=a)
    
    result.save(output_path+'/'+filename, format=None)
