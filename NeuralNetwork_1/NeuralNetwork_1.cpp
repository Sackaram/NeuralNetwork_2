#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "NeuralNetwork.h"

using namespace std;

pair<vector<vector<double>>, vector<vector<double>>> splitDataset(vector<vector<double>> dataset, double splitRatio) {
    int splitIndex = (int)(splitRatio * dataset.size());

    vector<vector<double>> trainSet(dataset.begin(), dataset.begin() + splitIndex);
    vector<vector<double>> testSet(dataset.begin() + splitIndex, dataset.end());

    return make_pair(trainSet, testSet);
}

vector<vector<double>> loadData(string filename) {
    vector<vector<double>> dataset;

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: could not open file " << filename << endl;
        exit(1);
    }

    string line;

    while (getline(file, line)) {
        vector<double> data;
        stringstream ss(line);
        double value;

        while (ss >> value) {
            data.push_back(value);
            ss.ignore();
        }

        int target = (int)data.back() - 1; // Convert target values from 1, 2, 3 to 0, 1, 2
        data.pop_back();
        data.push_back(target);

        dataset.push_back(data);
    }

    return dataset;
}

void train(NeuralNetwork& nn, vector<vector<double>> trainSet, int epochs, double learningRate) {
    srand(time(NULL));

    for (int epoch = 1; epoch <= epochs; epoch++) {
        double error = 0.0;

        for (vector<double> data : trainSet) {
            double* input = &data[0];
            int target = (int)data.back();

            nn.forward(input);
            nn.backward(input, target, learningRate);

            error += nn.getOutputLayer()[target];
        }

        if (epoch % 10 == 0) {
            cout << "Epoch " << epoch << " - Error: " << error << endl;
        }
    }
}

void test(NeuralNetwork& nn, vector<vector<double>> testSet) {
    int correct = 0;

    for (vector<double> data : testSet) {
        double* input = &data[0];
        int target = (int)data.back();

        int predicted = nn.predict(input);
        if (predicted == target) {
            correct++;
        }
    }

    double accuracy = (double)correct / testSet.size() * 100;
    cout << "Accuracy: " << accuracy << "%" << endl;
}

int main() {
    // Load dataset
    vector<vector<double>> dataset = loadData("iris.data");
    // Split dataset into training and testing sets
    pair<vector<vector<double>>, vector<vector<double>>> splitData = splitDataset(dataset, 0.7);
    vector<vector<double>> trainSet = splitData.first;
    vector<vector<double>> testSet = splitData.second;

    // Create neural network
    int inputSize = 4;
    int outputSize = 3;
    int hiddenSize = 12;

    auto start = std::chrono::high_resolution_clock::now();

    NeuralNetwork nn(inputSize, hiddenSize, outputSize);

    // Train neural network
    int epochs = 200;
    double learningRate = 0.1;

    train(nn, trainSet, epochs, learningRate);

    // Test neural network
    test(nn, testSet);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "Time to execute: " << duration.count() << " seconds" << std::endl;

    return 0;
}
