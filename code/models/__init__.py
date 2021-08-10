from models.sillnet import *
from models.sillnet_gtsrb import *

def get_model(name, class_train, class_test, feature_channel):
    model = _get_model_instance(name)
        
    if name is 'sillnet':
        model = model(nc=3, input_size = 64, class_train=class_train, class_test = class_test, extract_chn=[100, 150, 200, 150, 100, feature_channel], classify_chn = [100, 150, 200, 250, 300, 100], param1 = None, param2 = None, param3 = None, param4 = [150,150,150,150])
        print('Use sillnet with random initialization!')
    
    if name is 'sillnet_gtsrb':
        model = model(nc=3, input_size = 64, class_train=class_train, class_test = class_test, extract_chn=[150, 150, 150, 150, 150, feature_channel], classify_chn = [100, 150, 200, 150, 100, 100, 100], param1 = None, param2 = [150,150,150,150], param3 = [150,150,150,150], param4 = [150,150,150,150])
        print('Use sillnet_gtsrb with random initialization!')

    return model

def _get_model_instance(name):
    try:
        return {
            'sillnet' : SillNet,
            'sillnet_gtsrb' : SillNet_gtsrb,
        }[name]
    except:
        print('Model {} not available'.format(name))
