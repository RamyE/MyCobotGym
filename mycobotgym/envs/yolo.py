from ultralytics import YOLO



model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
