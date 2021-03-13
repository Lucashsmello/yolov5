from setuptools import setup

modules=['models.common','models.yolo', 'models.experimental','yolov5predictor', 'unpickler']
setup(name='ultralytics-yolov5',
    version='0.5',
#    description='',
    url='https://github.com/Lucashsmello/yolov5',
    maintainer='Lucas Mello',
    maintainer_email='lucashsmello@gmail.com',
    license='GPL-3.0',
    packages=['ultralytics/yolov5/utils'],
    py_modules=["ultralytics.yolov5.%s" % m for m in modules],
    install_requires=[
        "numpy >= 1.18.5",
        "torch >= 1.7.0",
        "torchvision >= 0.8.1"
    ])
