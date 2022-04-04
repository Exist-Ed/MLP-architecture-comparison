# MLP-regression

---

__USAGE:__ _python main.py_ <br>

Trains two multilayer perceptrons with the specified architectures on a data set generated on the basis of the passed
system of first-order differential equations and compares them according to the correctness
of the prediction of values.

# Contents of the code files

---


* main.py : &nbsp; &nbsp; &nbsp; implementing the main high-level logic
* DataSet.py : &nbsp; &nbsp; &nbsp; generating a data set for training and testing
* NN.py : &nbsp; &nbsp; &nbsp; implementation of a neural network, algorithms for its training and prediction of values
* visualization.py : &nbsp; &nbsp; &nbsp; algorithms for visualizing a data set, testing results, and comparing neural network architectures
* config.py : &nbsp; &nbsp; &nbsp; configuration file

# Configuration

---

You can specify the compared architectures of neural networks, the predicted system of first-order differential
equations and the characteristics of the data set for training and forecasting in the following file:

> ./config.py

# Installation

---

After launching the terminal, go to the directory where you want to install the program and perform the following manipulations:

1. Download the repository:
> git clone https://github.com/Exist-Ed/NLP-regression.git

2. Install the necessary dependencies:
> pip install -r requirements.txt

3. Use the program. Installation is complete




