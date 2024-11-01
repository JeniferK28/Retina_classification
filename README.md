# Retina_classification

## Requirements
Python 3 

* pytorch == 1.10.2
* torch_geometric 
* numpy >= 1.17.5 
* scipy >= 1.1.0 
* scikit-learn >= 0.22.2
* matplotlib 
* natsort 
* torchvision
* tensorboard
* itertools

## Files
We provide Python implementation for training the model, its evaluation and generate gradcam images.
* data_loader: Prepare data to input to the model
* retina_train.py: Train models
* train_pred.py: Train and prediction functions
* test_grad.py: Test data / save gradcam images
* utils.py: Save gradcam and overlap to input images
* Grad_cam.py: Calculate backpropagation and generate gradcam images
* plt_cm.py: Plot confusion matrix

- [x] Train model
```bash
python retina_train.py --k_folds 5 --train_data_path <path> --train_labels <path>
```

- [x] Test model
```bash
python test_grad.py --test_data_path <path> --test_labels <path> --model_file <path> --gradcam False
```

- [x] Save GradCAM images
```bash
python test_grad.py --test_data_path <path> --test_labels <path> --model_file <path> --target_layer 'layer4.1.conv1' --save_img_path <path> --gradcam True
```

## Results
* Training and validation performances for pre-trained RESNET, Vanilla RESNET and pre-trained RESNET with augmentation
<img src="retina_train.JPG" width=80% height=80%>
<img src="aug_retina.JPG" width=80% height=80%>

* Confusion matrix for pre-trained RESNET with augmentation
<img src="cm.jpeg" width=20% height=20%>

* Gradcam Results
<img src="grad_cam.JPG" width=80% height=80%>
