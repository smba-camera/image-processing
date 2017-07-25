import os
import image_processing.camera_model as Models

class Kitti:
    '''Loads all camera models from the data. They can be retrieved by using getCameraModel(camNum) and getVeloCameraModel()'''

    def __init__(self, path, date):
        self.camera_models = []
        self.velo_camera_model = None
        self.velo_extrinsic_model = None

        fullpath=os.path.join(path,date,date+'_calib',date)
        self.initCamtoCamParams(fullpath)
        self.initVelotoCamParams(fullpath, self.camera_models[0])

    def store_value_in_matrix(self, matrix, word, j):
        y=j//3
        x=j%3
        matrix[y][x]=float(word)

    def store_value_in_matrix_last_column(self, matrix, word, j):
        matrix[j][3]=float(word)

    def initCamtoCamParams(self, filepath):
        with open(os.path.join(filepath,'calib_cam_to_cam.txt')) as fp:
            lines = fp.readlines()
            ignored_lines = 2
            lines_per_camera = 8
            translation_line_index = 4
            rotation_line_index = 6
            intrinsic_line_index = 7


            cam_row_index = ignored_lines
            for cam_num in range(4):
                # get rotation
                rotation = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                line_index = cam_row_index + rotation_line_index
                words = lines[line_index].split()[1:]
                for j, word in enumerate(words):
                    self.store_value_in_matrix(rotation, word, j)

                # get translation
                line_index = cam_row_index + translation_line_index
                words = lines[line_index].split()[1:]
                translation = map(lambda w: float(w), words)

                # get intrinsic params
                line_index = cam_row_index + intrinsic_line_index
                words = lines[line_index].split()[1:]
                focal_length = float(words[0])
                optical_center_x = float(words[2])
                optical_center_y = float(words[6])

                # create camera model
                im = Models.IntrinsicModel(
                    focal_length=focal_length,
                    optical_center_x=optical_center_x,
                    optical_center_y=optical_center_y)
                em = Models.ExtrinsicModel(rotation=rotation, translationVector=translation)
                cm = Models.CameraModel(im=im, em=em)
                self.camera_models.append(cm)

                # next camera
                cam_row_index += lines_per_camera

        fp.close()

    def initVelotoCamParams(self,filepath, referenceCameraModel):
        rot_mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        trans_vect = [0,0,0]
        with open(os.path.join(filepath,'calib_velo_to_cam.txt')) as fp:
            for i,line in enumerate(fp):
                if i==1:
                    j=0
                    for word in line.split()[1:]:
                        self.store_value_in_matrix(rot_mat, word, j)
                        j+=1
                if i==2:
                    j=0
                    for word in line.split()[1:]:
                        trans_vect[j] = float(word)
                        j+=1
        fp.close()
        em = Models.ExtrinsicModel(rotation=rot_mat, translationVector=trans_vect)
        self.velo_extrinsic_model = em

        ems = referenceCameraModel.extrinsic_models
        im = referenceCameraModel.intrinsic_model
        extrinsic_models = [em]
        extrinsic_models.extend(ems)
        cm = Models.CameraModel(im=im, em=extrinsic_models)

        self.velo_camera_model = cm

    def getCameraModel(self, cam_num):
        return self.camera_models[cam_num]

    def getVeloCameraModel(self):
        return self.velo_camera_model

    def getVeloExtrinsicModel(self):
        return self.velo_extrinsic_model