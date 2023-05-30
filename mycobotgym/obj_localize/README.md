## Instructions
###Generating data
```
python generate_data.py --num_images=8000
```
**Note**
- Switch to the *mycobotgym/obj_localize/data directory*, and run the above script
- Training dataset will be collected under the folder *domain_rand*
  - Images will be stored under the folder *domain_rand/data*
  - Positions of target cube on each image will be recorded in the json file *domain_rand/pos_map.json*
###Running model
```
python train.py --num_epochs=30
```
**Note**
- Switch to the *mycobotgym/obj_localize/vision_model*, and run the above script
- Trained model weights will be stored under the directory *mycobotgym/obj_localize/vision_model/trained_models/domain_rand*