import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.misc as m
import random
import cv2

class gtsrb2gtsrbLoader(Dataset):

  def __init__(self, root, exp, split='train', is_transform=False, img_size=None, augmentations=None, prototype_sampling_rate=0.005):
    super().__init__()

    if split == 'train':
        self.proto_rate = prototype_sampling_rate
    else:
        self.proto_rate = 0.0
        
    self.inputs = []
    self.targets = []
    self.class_names = []
    self.split = split
    self.img_size = img_size
    self.is_transform = is_transform
    self.augmentations = augmentations
    self.mean = np.array([125.00, 125.00, 125.00]) # average intensity
    
    if split == 'train':
        self.split = 'GTSRB'
        self.n_classes = 43 
        self.tr_class = np.array([1,2,3,4,5,7,8,9,10,11,12,13,14,15,17,18,25,26,31,33,35,38])
    elif split == 'test':
        self.split = 'GTSRB'
        self.n_classes = 43 
        self.te_class = np.array([0,6,16,19,20,21,22,23,24,27,28,29,30,32,34,36,37,39,40,41,42])
    
    self.root = root
    #exp  = exp + '/exp_gtsrb/'
    self.dataPath = root + exp + '/' + self.split + '_impaths_all.txt'
    self.labelPath = root + exp + '/' + self.split + '_imclasses_all.txt'

    f_data = open(self.dataPath,'r')
    f_label = open(self.labelPath,'r')
    data_lines = f_data.readlines()
    label_lines = f_label.readlines()

    for i in range(len(data_lines)):
        if split == 'train':
            label = int(label_lines[i].split()[0])  
            if label-1 in self.tr_class.tolist():
                self.inputs.append(root+data_lines[i][0:-1])
                self.targets.append(label) # label: [road class, wet/dry, video index]
        elif split == 'test':
            label = int(label_lines[i].split()[0])
            if label-1 in self.te_class.tolist():
                self.inputs.append(root+data_lines[i][0:-1])
                self.targets.append(label) # label: [road class, wet/dry, video index]
    
    classnamesPath = root + exp + '/' + self.split + '_classnames.txt'
    f_classnames = open(classnamesPath, 'r')
    data_lines = f_classnames.readlines()
    for i in range(len(data_lines)):
        self.class_names.append(data_lines[i][0:-1])

    self.n_classes = 43
    assert(self.n_classes == len(self.class_names))

    print('GTSRB %d classes'%(len(self.class_names)))
    print('Load GTSRB %s: %d samples'%(split, len(self.targets)))


  def __len__(self):
    return len(self.inputs)


  def __getitem__(self, index):
    img_path = self.inputs[index]
    gt = self.targets[index]
    gt = torch.ones(1).type(torch.LongTensor)*gt

    # Load images and templates. perform augmentations
    img = m.imread(img_path)
    img = np.array(img, dtype=np.uint8)
    template = m.imread(self.root + self.split + '/template_ordered/%02d.jpg'%(gt))
    template = np.array(template, dtype=np.uint8)

    if random.random() < self.proto_rate:
        img = np.copy(template)

    if self.augmentations is not None:
        img, template = self.augmentations(img, template)

    if self.is_transform:
        img = self.transform(img)
        template = self.transform(template)
    gt = gt-1
    return img, gt, template
    
  def transform(self, img):
    img = img.astype(np.float64)
    #img -= self.mean
    if self.img_size is not None:
      img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation = cv2.INTER_AREA)
    # Resize scales images from 0 to 255, thus we need
    # to divide by 255.0

    
    img = img.astype(float) / 255.0
    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    
    return img


  def load_template(self, target, augmentations=None):

    # if augmentation is not specified, use self.augmentations. Unless use input augmentation option.
    if augmentations is None:
        augmentations = self.augmentations
    img_paths = []
    
    for id in target:
        img_paths.append(self.root + self.split + '/template_ordered/%02d.jpg'%(id+1))

    target_img = []
    target_template = []
    for img_path in img_paths:
        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        if augmentations is not None:
            img, template = augmentations(img, img)
        if self.transform:
            img = self.transform(img)
            template = self.transform(template)

        target_img.append(img)
        target_template.append(template)

    return torch.stack(target_img, dim=0), torch.stack(target_template, dim=0)