import glob,os
import numpy as np

P_Rect=[[0,0,0,0],[0,0,0,0],[0,0,0,0]]
K_Rect=[[0,0,0],[0,0,0],[0,0,0]]
R_Rect=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]
T_CamVelo=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]

def storeValuePRect(word,j):
    y=j//4
    x=j%4
    P_Rect[y][x]=float(word)

def storeValueRRect(word,j):
    y=j//3
    x=j%3
    R_Rect[y][x]=float(word)

def storeValueRot_CamVelo(word,j):
    y=j//3
    x=j%3
    T_CamVelo[y][x]=float(word)

def storeValueTrans_CamVelo(word,j):
    T_CamVelo[j][3]=float(word)

def initCamtoCamParams(CamNum):
    with open('calib_cam_to_cam.txt') as fp:
        for i,line in enumerate(fp):
            if i==(CamNum+1)*8:
                j=0
                for word in line.split(' ',1)[1].split():
                    storeValueRRect(word,j)
                    j+=1

            if i==1+(CamNum+1)*8:
                j=0
                for word in line.split(' ',1)[1].split():
                    storeValuePRect(word,j)
                    j+=1
    fp.close()


def initVelotoCamParams():
    with open('calib_velo_to_cam.txt') as fp:
        for i,line in enumerate(fp):
            if i==1:
                j=0
                for word in line.split(' ',1)[1].split():
                    storeValueRot_CamVelo(word,j)
                    j+=1
            if i==2:
                j=0
                for word in line.split(' ',1)[1].split():
                    storeValueTrans_CamVelo(word,j)
                    j+=1
    fp.close()

def initialize(path,date,CamNum=0):
    os.chdir(path+'\\'+date+'\\'+date+'_calib\\'+date)
    initCamtoCamParams(CamNum)
    initVelotoCamParams()

def transform3DtoImageCoordinates(D3Coordinates):
    temp=np.dot(P_Rect,R_Rect)
    temp=np.dot(temp,T_CamVelo)
    return np.dot(temp,D3Coordinates)
    #return np.dot(P_Rect,D3Coordinates)

def getImageCoordinates(D3Coordinates):
    temp=transform3DtoImageCoordinates(D3Coordinates)
    print ([temp[0] / temp[2], temp[1] / temp[2]])
    return [temp[0]/temp[2],temp[1]/temp[2]]