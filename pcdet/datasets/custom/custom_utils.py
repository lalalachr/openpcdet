import numpy as np

def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects

def cls_type_to_id(cls_type):
    # type_to_id = {'pedestrian': 1, 'ech': 2, 'container_truck': 3}
    type_to_id = {'ech': 1, 'truck': 2}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])
        self.occlusion = float(label[2])
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        # self.h = float(label[8])
        # self.w = float(label[9])
        # self.l = float(label[10])
        self.h = float(label[10])
        self.w = float(label[9])
        self.l = float(label[8])
        # self.loc = np.array((float(label[13]), -float(label[11]), -float(label[12])), dtype=np.float32)
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[14])
        # self.ry = -(float(label[14])+np.pi/2)
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = 'Easy'
        self.level = 0
        # self.box3d = np.array((float(label[13]), -float(label[11]), -float(label[12]),
        #                        float(label[10]), float(label[9]), float(label[8]),
        #                        float(label[14])), dtype=np.float32)
        self.box3d = np.array((float(label[11]), float(label[12]), float(label[13]),
                               float(label[10]), float(label[9]), float(label[8]),
                               float(label[14])), dtype=np.float32)
