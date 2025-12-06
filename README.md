# Wind-PG_GAN
Using physics-guided generative adversarial network (PG-GAN) to recontruct sea surface wind field
## Dependence
Tensorflow 2.0
## Weight
The weights directory contains links to the pre-trained model checkpoints stored on Google Drive.
You can download the weights from the link and place them in the appropriate folder before running inference.
## Code
The inference script is located in code/PG_GAN/. Other components of the project will be added later.
To reproduce the results with the pre-trained model:
```
python main.py plot --scenes_file=<file>
```
The plotting script is located in code/plot/. A link to result files generated with the pre-trained weights is also provided for reference.
