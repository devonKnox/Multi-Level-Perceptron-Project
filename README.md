# **Multi-Layer Perceptron Project**

This project implements a Multi-Layer Perceptron (MLP) neural network in Java from scratch, without using any external machine learning libraries. The MLP was used to solve the following three problems:

- **XOR Function**: 2 inputs (1 or 0 for each input), 1 output (1 or 0)
- **Sine Function Approximation**: given input vector [x1, x2, x3, x4], the output should be sin(x1-x2+x3-x4)
- **Letter Recognition**: using the UCI Letter Recognition dataset (see http://archive.ics.uci.edu/ml/datasets/Letter+Recognition) where the first entry of each line is the letter to be recognized (the output) and the proceeding numbers are attributes extracted from images of each letter (the inputs), the MLP must accurately predict letters in the alphabet

---

## **Table of Contents**

1. [Project Structure](#project-structure)
2. [Understanding the Code](#understanding-the-code)
   - [MLP Class](#mlp-class)
   - [Key Methods](#key-methods)
   - [Activation Functions](#activation-functions)
3. [Results](#results)

---

## **Project Structure**

- **ActivationFunctionType.java**: Enum class defining activation function types (`SIGMOID`, `LINEAR`, `SOFTMAX`). Each activation function is best used for a specific problem.

- **MLP.java**: The main Multi-Layer Perceptron class implementing the neural network.

- **TrainingExample.java**: Class to hold input-output pairs for training.

- **XORExperiment.java**: Implements the XOR problem experiment by training the MLP. Running this program creates the text file **XORExperimentResults.txt**, which displays the results and verifies that the MLP correctly predicts each input.

- **SineExperiment.java**: Implements the sine function approximation experiment by generating 500 vectors, each with four components between -1 and 1, and trains the MLP to output sin(x1-x2+x3+x4) on 400 of the examples. The last 100 input vectors are then tested on the newly-trained MLP. Running this program outputs the results of this experiment in the file **SineExperimentResults.txt**.

- **LetterRecognitionExperiment.java**: Implements the letter recognition experiment by training the MLP on the UCI Letter Recognition Dataset. The dataset is split into a training set containing 80% of the data and a testing set with the remaining 20%. The MLP is configured with 16 inputs (corresponding to the dataset attributes), 40 hidden units, and 26 outputs (one for each letter of the alphabet). The model is trained for 2000 epochs using the softmax activation function for the output layer. After training, the program evaluates the MLP on the test set, calculates the classification accuracy, and outputs the results to the file **LetterRecognitionExperimentResults.txt**.

- **letter-recognition.data**: Dataset file for the letter recognition experiment.

- **README.md**: This readme file.

---

## **Understanding the Code**

### **MLP Class**
The `MLP` class implements a neural network with the following features:

- **Flexible Architecture**: Supports any number of inputs, hidden units, and outputs, allowing for customizable network structures.
- **Activation Functions**: Configurable activation functions for the output layer, including **sigmoid**, **linear**, and **softmax**, tailored to different types of tasks.

### **Key Methods**
- `forward(double[] input)`: Performs forward propagation, calculating the activations for all layers based on the input.
- `backward(double[] input, double[] target, double learningRate)`: Computes gradients during backpropagation and updates weights incrementally.
- `updateWeights(double learningRate)`: Applies accumulated weight updates to the network's weights after each batch or epoch.
- `randomizeWeights()`: Initializes the weights to small random values, ensuring the network starts with a good foundation for training.

### **Activation Functions**
The activation functions are defined in `ActivationFunctionType.java` and are utilized in the `MLP` class to adjust the network's behavior:

- **SIGMOID**: Suitable for binary classification tasks, where outputs are probabilities between 0 and 1.
- **LINEAR**: Used for regression tasks, where outputs can take any continuous value.
- **SOFTMAX**: Ideal for multi-class classification tasks, where outputs represent probabilities across multiple classes.

---
## **Results**

The project successfully implements a flexible Multi-Layer Perceptron (MLP) in Java, capable of supporting variable inputs, hidden layers, and outputs. Here are the specific results for each experiment:

- **XOR Experiment**: The MLP was configured with 2 inputs, 4 hidden units, and 1 output. By selecting an optimal learning rate and training for 5000 epochs, the model accurately predicted the XOR function outputs, demonstrating the ability to learn a nonlinear relationship.
- **Sine Function Approximation**: With 4 inputs, 5 hidden units, and 1 output, the MLP was trained on 400 examples to approximate the function $ \sin(x_1 - x_2 + x_3 - x_4) $. By fine-tuning the learning rate and leveraging a linear activation function for the output, the model achieved low training and test errors, effectively generalizing to unseen data.
- **Letter Recognition**: Using the UCI Letter Recognition Dataset, the MLP was configured with 16 inputs, 40 hidden units, and 26 outputs. By training for 2000 epochs with a softmax output layer, the model achieved high classification accuracy, demonstrating its ability to handle multi-class classification tasks.

Across all experiments, adjustments to learning rates, batch sizes, and the number of hidden units allowed the MLP to achieve strong performance, effectively learning and generalizing in a variety of scenarios.
