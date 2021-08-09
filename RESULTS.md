### Plant/Weed Segmentation

Goal of this project is to prototype and showcase a segmentation model on a custom plant / weed dataset.

-------------------------

#### Running The Code

* Download the dataset and place it in the dataset folder as shown below. 
* Directory Structure
  * dataset
    * __sugarbeet_weed_dataset__  <-----( place the dataset folder here )
  * examples
  * src
  * trained model
* Create Python Virtual Env and install 'requirements.txt'
  ```
  cd coding_task
  
  python3 -m venv ./envs 
  
  source ~/envs/venv/bin/activate
  
  pip3 install -r requirements.txt
  ```
* To run __Training__ :
  ```
  python3 seg_trainer.py
  ```
* To run __Inference__ :
  ```
  python3 seg_inference.py
  ```
-------------------------

#### Design Choices

* Pytorch along with Pytorch Lightning [1] is used for prototyping the model. Pytorch Lightning is used to scale the
  model training on multiple GPUs/TPUs.
* Due to time / computational constraints, the model was trained only for 10 epochs with a dataset sample size of 1000
  images and the results obtained can be further improved. Training PC : i5 4-core, Nvidia 1050Ti, 8Gb RAM
* The code base consists of three different models which was used for prototyping. The final training and inference was
  only done on Pytorch Mask-RCNN:
    * Pytorch Mask-RCNN [3] : Outputs bounding box, segmentation mask, keypoint
    * Custom UNet model : Custom UNet model written which outputs only segmentation mask but has no classification,
      regression head.
    * segmentation-models-pytorch [8]: Contains many pretrained segmentation models for simple training.
* The evaluation metrics can be found using modified coco eval utilities [9].

-------------------------

#### Code

* 'src/seg_inference.py' has the main Inference and Visualization code.
* 'src/seg_trainer.py' contains the Pytorch Lightning Module that can be used for training the segmentation model.
* 'src/utils/*.py' contains all utilities functions required for training, inference, evaluation and visualization loop.
* 'src/config/custom_config.py' contains the modified config file used by us.
    * The image has been resized to (224,224) for easier prototyping and training.
    * The input is normalized according to Imagenet preprocessing values.
* Along with the 3 classes in the dataset, an extra is added for considering the blank parts (due to rotation / skew
  transformations) of the images.

-------------------------

#### Results

* Due to computational and time constraints the model has not been trained to convergence.

-------------------------

#### Extra Ideas

* Since the data classes {Soil, SugarBeet, Weed, Blank} are unbalanced, methods like adding weightage to the
  CrossEntropyLoss etc. can be used to improve training.
* There are other types of augmentations like adding noise, brightness, alpha blend can be added to improve robustness
  of the trained model.
* Prototype end-end models trained only on our dataset or try other segmentation model architectures.
* Proper metrics for testing quality of segmentation [7] can be used.
* Frameworks like Detectron2 can be used to train the model and perform evaluation.  

-------------------------

#### References

1. https://www.pytorchlightning.ai/
2. https://github.com/ternaus/cloths_segmentation
3. https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
4. https://pytorch.org/vision/master/auto_examples/plot_visualization_utils.html
5. https://colab.research.google.com/drive/11FN5yQh1X7x-0olAOx7EJbwEey5jamKl?usp=sharing#scrollTo=M6ME2_rvGv_G
6. https://debuggercafe.com/instance-segmentation-with-pytorch-and-mask-r-cnn/
7. https://www.jeremyjordan.me/evaluating-image-segmentation-models/
8. https://segmentation-modelspytorch.readthedocs.io/en/latest/
9. https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
10. https://detectron2.readthedocs.io/en/latest/index.html

-------------------------

#### Author

Name : Anirudh NJ           
Gmail : [anijaya9@gmail.com](anijaya9@gmail.com)        
Github : [https://github.com/njanirudh/](https://github.com/njanirudh/)