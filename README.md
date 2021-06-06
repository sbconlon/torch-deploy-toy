# Torch Delploy Toy

This repository is a simple project intended to be a proof-of-concept to aide in the deployment of the Exatrkx inference pipeline inside the ACTS framework. A simple MNIST classifier is trained and scripted in the `train.py` file. Then, the C++ project can be compiled and ran to classify random images in parallel. Note, the TBB parallel for loop was used because it is identical to that of the event level loop in the ACTS sequencer object (https://github.com/sbconlon/acts/blob/master/Examples/Framework/src/Framework/Sequencer.cpp#L260) 
