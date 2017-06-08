import os
import image_processing.camera_model.Models as Models

class Kitti:
    def __init__(self):
        self.focal_length=0
        self.optical_center_x = 0  # optical center on the image is on coordinates (0,0)
        self.optical_center_y = 0

    def store_value_in_matrix(self, matrix, word, j):
        y=j//3
        x=j%3
        matrix[y][x]=float(word)

    def store_value_in_matrix_last_column(self, matrix, word, j):
        matrix[j][3]=float(word)

    def initCamtoCamParams(self, filepath, CamNum):
        R_Rect = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        with open(filepath+'\calib_cam_to_cam.txt') as fp:
            for i,line in enumerate(fp):
                if i==(CamNum+1)*8:
                    j=0
                    for word in line.split(' ',1)[1].split():
                        self.store_value_in_matrix(R_Rect, word, j)
                        j+=1

                if i==1+(CamNum+1)*8:
                    j=0
                    for word in line.split(' ',1)[1].split():
                        if j==0:
                            self.focal_length=float(word)
                        if j==2:
                            self.optical_center_x=float(word)
                        if j==6:
                            self.optical_center_y=float(word)
                        j+=1
        fp.close()
        self.extrinsic_Model=Models.ExtrinsicModel(R_Rect)

    def initVelotoCamParams(self,filepath):
        T_CamVelo = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
        with open(filepath+'\calib_velo_to_cam.txt') as fp:
            for i,line in enumerate(fp):
                if i==1:
                    j=0
                    for word in line.split(' ',1)[1].split():
                        self.store_value_in_matrix(T_CamVelo, word, j)
                        j+=1
                if i==2:
                    j=0
                    for word in line.split(' ',1)[1].split():
                        self.store_value_in_matrix_last_column(T_CamVelo, word, j)
                        j+=1
        fp.close()
        self.Cam_to_Velo_Model = Models.ExtrinsicModel(T_CamVelo)

    def initialize(self,path,date,CamNum=0,):

        fullpath=os.path.join(path,date,date+'_calib',date)
        self.initCamtoCamParams(fullpath,CamNum)
        self.initVelotoCamParams(fullpath)
        self.intrinsic_Model=Models.IntrinsicModel(focal_length=self.focal_length,optical_center_x=self.optical_center_x,optical_center_y=self.optical_center_y)
        self.camera_Model=Models.CameraModel(self.intrinsic_Model,[self.extrinsic_Model,self.Cam_to_Velo_Model])
        return self.camera_Model

    def getCameraModel(self):
        return self.camera_Model