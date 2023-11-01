Real-Time Deep Optical Flow Estimation
-----------------------
This code is a PyTorch implementation of the model described in the Master’s Thesis “Real-Time Deep Optical Flow Estimation”.
It is based on the code of GitHub user MJITG for HITNet, which can be found here:
https://github.com/MJITG/PyTorch-HITNet-Hierarchical-Iterative-Tile-Refinement-Network-for-Real-time-Stereo-Matching
See the thesis for more details.

Make sure the KITTI dataset is placed in the KITTI folder under datasets.
Generate the ground truth slant parameters using the code in the slants folder.
Feature extraction weights of a fully trained HITNet are stored in the experiment_0 checkpoint file.
Run the model with kitti.sh.

Model overview
-----------------------
<p float="left">
  <img src="/img/Overview.drawio-2.png"/>
</p>

Feature extraction
-----------------------
<p float="left">
  <img src="/img/Feature extraction.drawio-2 copy-horizontal.png"/>
</p>

Initialization
-----------------------
<p float="left">
  <img src="/img/Initialization.drawio-2.png"/>
</p>

Refinement
-----------------------
<p float="left">
  <img src="/img/Refinement.drawio-2.png"/>
</p>

ResNet
-----------------------
<p float="left">
  <img src="/img/ResNet.drawio-2.png"/>
</p>

Results
-----------------------
<p float="left">
  <img src="/img/bilinear_example.png"/>
</p>