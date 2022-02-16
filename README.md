# AdversarialAblation
This repo contains the implementation of the ablation techniques described in the paper Adversarial Input Ablation for Audio-Visual Learning.

### Pre-trained models
| Ablation technique  | R@10          |
| -------------       |:-------------:| 
| [Frame-based](https://drive.google.com/file/d/1HGrDrhu08EAboYU6BmPsYb2d9e-G5_Xu/view?usp=sharing)      | 0.859         | 
| [Random](https://drive.google.com/file/d/1U438zkyv4kP9JpgOX59K2mm-x2xkDDao/view?usp=sharing)           | 0.839         |
| [Oracle-based](https://drive.google.com/file/d/1aYJEykHK90fSvaBzYFHxvH13_CNITHB6/view?usp=sharing)      | 0.856         | 
### Data
We use the SpokenCOCO dataset, which can be found [here](https://groups.csail.mit.edu/sls/downloads/placesaudio/index.cgi). 

### Model
The baseline model used in this paper is the ResDAVEnet-VQ model. More information on this model and an implementation can be found at https://github.com/wnhsu/ResDAVEnet-VQ.

