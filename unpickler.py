import pickle
from pickle import *
class moduleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # print("find_class(%s,%s)" % (module, name))
        mapp = {'models': "ultralytics.yolov5.models"}
        spl = module.split('.', 1)
        if(spl[0] in mapp):
            module = mapp[spl[0]]+'.'+spl[1]

        return super().find_class(module, name)

Unpickler=moduleUnpickler