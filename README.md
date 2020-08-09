# My MSc Thesis project on Anomaly Detection in Network Traffic with the use of Multivariate State Machines
This repository contains all the code related to the work I have done for my MSc thesis during the last 9 months of my master studies in Computer Science at TU Delft. In this project, an **end-to-end detection pipeline** for identifying malicious (or anomalous) behavior in NetFlow data based on **the extraction of multivariate behavioral profiles solely from benign traffic** is implemented. In particular, the functionality of the proposed detection system is founded on the assumption that, if models are extracted from the normal operation of a monitored system, then any behavior rejected from these models is considered an anomaly, thus it could be potentially associated to malicious activity. A high level illustration of the detection process followed in this work can be seen below.

<p align="center">
<img src="https://github.com/SereV94/MasterThesis/blob/master/images/detection_structure.png" height="300" width="500">
</p>

In brief, state-of-the-art automata learning algorithms are used to **extract multivariate state machines from the benign NetFlow traces** of certain network entities, like hosts and connections. As a result, each state machine can be perceived as a communication profile capturing the benign behavior of the corresponding network entity, with each state of the derived state machines acting as a temporal cluster of the general behavior of the corresponding network entity. Subsequently, the benign traces are replayed on each corresponding state machine, and **an anomaly detection model (LOF, Isolation Forest, Gaussian KDE)** is learnt in each state. As a result, each state can be used as a detection predictor when unseen NetFlow traces are replayed on the associated state machine. Finally, each fitted multivariate state machine is utilized as a benign communication profile providing predictions on unseen sets of NetFlow traces, with the benignity of each set depending on the extent of its match to the extracted benign profiles.

The problem of detecting anomalies in the recorded NetFlow traffic is dealt as a classification problem, with the two classes being the benign and malicious network entitities. Thus, the components of the designed detection system are structured in two phases: **the training, and the testing phases**. A high level presentation of that structure can be seen as follows:

<p align="center">
<img src="https://github.com/SereV94/MasterThesis/blob/master/images/my_pipeline.png" height="200" width="600">
</p>

Before providing more information regarding the intrinsics of these phases, it shall be mentioned that most components of the proposed system are implemented in **Python**, while **Jupyter notebooks** have been used for visualization purposes. The only component of the system that does not conform to this pattern regards the multivariate state machine inference process, for which **flexfringe** was used. [Flexfringe](https://bitbucket.org/chrshmmmr/dfasat/src/master/) is a tool implemented in **C++**, thus a Python wrapper was developed to invoke it from a Python script.

## The training phase
* Initial data preparation
  1. Group the provided flows
  2. Trace extraction through the use of dynamic windows
* Benign behavioral profiles' extraction <p align="center"> <img src="https://github.com/SereV94/MasterThesis/blob/master/images/train_pipeline.png" height="200" width="600"> </p>
  1. Multivariate FSM inference
  2. Replay the traces on the extracted FSMs
  3. Fit detection model on each state <p align="center"> <img src="https://github.com/SereV94/MasterThesis/blob/master/images/sample_dfa_netflow.png" height="250" width="700"> </p>
  4. Populate the "database" of benign profiles

## The testing phase
* Initial data preparation
  1. Group the provided flows
  2. Trace extraction through the use of dynamic windows
* Anomalies identification <p align="center"> <img src="https://github.com/SereV94/MasterThesis/blob/master/images/test_pipeline.png" height="150" width="600"> </p>
  1. Replay the testing traces on each of the extracted models
  2. Make state-local predictions
  3. Aggregate and collect the weighted state-local predictions
  4. Provide the final detection verdict by comparing the benignity score of the testing entity to the validation threshold
