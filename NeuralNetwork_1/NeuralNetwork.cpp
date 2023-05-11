#include "NeuralNetwork.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
    this->inputSize = inputSize;
    this->hiddenSize = hiddenSize;
    this->outputSize = outputSize;

    // Initialize weight matrices and bias vectors
    weightsInputHidden = new double* [hiddenSize];
    for (int i = 0; i < hiddenSize; i++) {
        weightsInputHidden[i] = new double[inputSize];
        for (int j = 0; j < inputSize; j++) {
            weightsInputHidden[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }

    weightsHiddenOutput = new double* [outputSize];
    for (int i = 0; i < outputSize; i++) {
        weightsHiddenOutput[i] = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            weightsHiddenOutput[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }

    biasHidden = new double[hiddenSize];
    for (int i = 0; i < hiddenSize; i++) {
        biasHidden[i] = ((double)rand() / RAND_MAX) - 0.5;
    }

    biasOutput = new double[outputSize];
    for (int i = 0; i < outputSize; i++) {
        biasOutput[i] = ((double)rand() / RAND_MAX) - 0.5;
    }

    hiddenLayer = new double[hiddenSize];
    outputLayer = new double[outputSize];
}

NeuralNetwork::~NeuralNetwork() {
    // Free memory allocated for weight matrices and bias vectors
    for (int i = 0; i < hiddenSize; i++) {
        delete[] weightsInputHidden[i];
    }
    delete[] weightsInputHidden;

    for (int i = 0; i < outputSize; i++) {
        delete[] weightsHiddenOutput[i];
    }
    delete[] weightsHiddenOutput;

    delete[] biasHidden;
    delete[] biasOutput;
    delete[] hiddenLayer;
    delete[] outputLayer;
}

void NeuralNetwork::matrixMultiply(double** A, double** B, double** C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void NeuralNetwork::forward(double* input) {
    // Calculate values for hidden layer
    for (int i = 0; i < hiddenSize; i++) {
        double activation = biasHidden[i];
        for (int j = 0; j < inputSize; j++) {
            activation += weightsInputHidden[i][j] * input[j];
        }
        hiddenLayer[i] = 1.0 / (1.0 + exp(-activation));
    }
    // Calculate values for output layer
    for (int i = 0; i < outputSize; i++) {
        double activation = biasOutput[i];
        for (int j = 0; j < hiddenSize; j++) {
            activation += weightsHiddenOutput[i][j] * hiddenLayer[j];
        }
        outputLayer[i] = 1.0 / (1.0 + exp(-activation));
    }
}

void NeuralNetwork::backward(double* input, int target, double learningRate) {
    // Calculate error for output layer
    double* outputError = new double[outputSize];
    for (int i = 0; i < outputSize; i++) {
        outputError[i] = (target == i ? 1.0 : 0.0) - outputLayer[i];
    }
    // Calculate error for hidden layer
    double* hiddenError = new double[hiddenSize];
    for (int i = 0; i < hiddenSize; i++) {
        double error = 0.0;
        for (int j = 0; j < outputSize; j++) {
            error += outputError[j] * weightsHiddenOutput[j][i];
        }
        hiddenError[i] = error * hiddenLayer[i] * (1.0 - hiddenLayer[i]);
    }

    // Update weights and biases for output layer
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < hiddenSize; j++) {
            weightsHiddenOutput[i][j] += learningRate * outputError[i] * hiddenLayer[j];
        }
        biasOutput[i] += learningRate * outputError[i];
    }

    // Update weights and biases for hidden layer
    for (int i = 0; i < hiddenSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            weightsInputHidden[i][j] += learningRate * hiddenError[i] * input[j];
        }
        biasHidden[i] += learningRate * hiddenError[i];
    }

    delete[] outputError;
    delete[] hiddenError;
}

int NeuralNetwork::predict(double* input) {
    forward(input);
    int predicted = -1;
    double maxOutput = -INFINITY;

    for (int i = 0; i < outputSize; i++) {
        if (outputLayer[i] > maxOutput) {
            maxOutput = outputLayer[i];
            predicted = i;
        }
    }

    return predicted;
}
int NeuralNetwork::getInputSize() {
    return inputSize;
}

int NeuralNetwork::getHiddenSize() {
    return hiddenSize;
}

int NeuralNetwork::getOutputSize() {
    return outputSize;
}

double** NeuralNetwork::getWeightsInputHidden() {
    return weightsInputHidden;
}

double** NeuralNetwork::getWeightsHiddenOutput() {
    return weightsHiddenOutput;
}

double* NeuralNetwork::getBiasHidden() {
    return biasHidden;
}

double* NeuralNetwork::getBiasOutput() {
    return biasOutput;
}

double* NeuralNetwork::getHiddenLayer() {
    return hiddenLayer;
}

double* NeuralNetwork::getOutputLayer() {
    return outputLayer;
}