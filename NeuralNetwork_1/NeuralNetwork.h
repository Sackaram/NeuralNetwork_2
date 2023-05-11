#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

class NeuralNetwork {
private:
    int inputSize;
    int hiddenSize;
    int outputSize;

    double** weightsInputHidden;
    double** weightsHiddenOutput;
    double* biasHidden;
    double* biasOutput;
    double* hiddenLayer;
    double* outputLayer;

    void matrixMultiply(double** A, double** B, double** C, int N);

public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize);
    ~NeuralNetwork();

    int predict(double* input);
    int getInputSize();
    int getHiddenSize();
    int getOutputSize();
    void forward(double* input);
    void backward(double* input, int target, double learningRate);
    double** getWeightsInputHidden();
    double** getWeightsHiddenOutput();
    double* getBiasHidden();
    double* getBiasOutput();
    double* getHiddenLayer();
    double* getOutputLayer();
};

#endif
