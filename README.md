# interativeoutputpretraining
Overview
This repository contains implementations of four distinct SOMnet-based models, each built on different neural network architectures and compatible with specific TensorFlow versions. Below is detailed information on each model, including its required TensorFlow version, directory structure, and execution commands.
Model Details & Execution Guide
Each model is organized in a dedicated directory. Before running any script, ensure you have installed the correct TensorFlow version for the target model to avoid compatibility issues.
Model Name	Required TensorFlow Version	Directory	Core Components
FC-SOMnet	1.15.5	FC-SOMnet	Plain FC One, Plain FC Two, FC-SOMnet
CNN-SOMnet	2.5.0	CNN-SOMnet	Plain CNN One, Plain CNN Two, CNN-SOMnet
Trans-SOMnet	2.5.0	Trans-SOMnet	Plain Transformer One, Plain Transformer Two, TranSOMnet
CLIPA-SOMnet	2.5.0	CLIPA-SOMnet	Plain CLIP Adapter One, Plain CLIP Adapter Two, CLIPASOMnet
1. FC-SOMnet
Prerequisite
Install TensorFlow 1.15.5 first:
bash
pip install tensorflow==1.15.5
Execution Steps
Navigate to the FC-SOMnet directory:
bash
cd FC-SOMnet
Run the required component:
For Plain FC One:
bash
python plainone.py
For Plain FC Two:
bash
python plaintwo.py
For FC-SOMnet:
bash
python FCSOMnet.py
2. CNN-SOMnet
Prerequisite
Install TensorFlow 2.5.0 first:
bash
pip install tensorflow==2.5.0
Execution Steps
Navigate to the CNN-SOMnet directory:
bash
cd CNN-SOMnet
Run the required component:
For Plain CNN One:
bash
python plainone.py
For Plain CNN Two:
bash
python plaintwo.py
For CNN-SOMnet:
bash
python FCSOMnet.py
3. Trans-SOMnet
Prerequisite
Install TensorFlow 2.5.0 first:
bash
pip install tensorflow==2.5.0
Execution Steps
Navigate to the Trans-SOMnet directory:
bash
cd Trans-SOMnet
Run the required component:
For Plain Transformer One:
bash
python plainone.py
For Plain Transformer Two:
bash
python plaintwo.py
For TranSOMnet:
bash
python TranSOMnet.py
4. CLIPA-SOMnet
Prerequisite
Install TensorFlow 2.5.0 first:
bash
pip install tensorflow==2.5.0
Execution Steps
Navigate to the CLIPA-SOMnet directory:
bash
cd CLIPA-SOMnet
Run the required component:
For Plain CLIP Adapter One:
bash
python plainone.py
For Plain CLIP Adapter Two:
bash
python plaintwo.py
For CLIPASOMnet:
bash
python CLIPASOMnet.py
Notes
TensorFlow Version Compatibility: FC-SOMnet requires TensorFlow 1.x (1.15.5), while the other three models rely on TensorFlow 2.x (2.5.0). Mixing versions may cause runtime errors.
Directory Navigation: Always ensure you are in the correct model directory before executing scripts.
Dependency Installation: If additional dependencies are required (e.g., NumPy, Pandas), install them via pip as needed.
