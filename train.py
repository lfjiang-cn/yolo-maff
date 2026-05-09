import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('')

    # model.load('yolov8n.pt')

    model.train(data=r'ultralytics/cfg/datasets/tt100k.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                single_cls=False,  
                batch=4,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD', 
                # resume='runs/train/exp21/weights/last.pt', 
                amp=True,  
                project='runs/train',
                name='v8s',
                )