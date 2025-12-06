# Wind-PG_GAN
Using physics-guided generative adversarial network (PG-GAN) to recontruct sea surface wind field
## Dependence
Tensorflow 2.0
## Weight
The weight folder contains the trained model weight files for all cases. 
## Code
The infer program is located in the code/PG_GAN folder, and the rest of the project will be added later. 
To reproduce the results with the pre-trained model:
    python main.py plot --scenes_file=<file>
