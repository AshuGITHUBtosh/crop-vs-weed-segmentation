# Crop–Weed Instance Segmentation for Laser Weeding
---
### This repository implements a hybrid perception pipeline for instance-level weed segmentation suitable for plant-level laser actuation in agricultural environments.
The system is designed to safely separate crops from weeds and output individual weed instances, each with its own mask and centroid, enabling precise and targeted laser-based weed removal.

---
## System Requirements

- ultralytics
- python-dotenv

---

## Install dependencies

```
pip install ultralytics
pip install python-dotenv

 ```
## Setting Paths
### In the .env file there are 3 paths
- Path of the trained model (best.pt)
- path of the input image
- path of the output folder
### Change the paths according to yourr system before running the code.

---

# Initializing the Code
``` python test.py ```

### The output will be stored in the "output" folder 
