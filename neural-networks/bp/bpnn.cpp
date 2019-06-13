#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include "bpnn.h"

using namespace std;

BPNN::BPNN(int input_length, int hidden_length, int output_length, 
        double learn_rate)
    : L1(input_length), L2(hidden_length), L3(output_length), lr(learn_rate)
{

    weight1 = new double*[L1];
    for (int i = 0; i < L1; i++)
        weight1[i] = new double[L2];

    weight2 = new double*[L2];
    for (int i = 0; i < L2; i++)
        weight2[i] = new double[L3];

    threshold1 = new double[L2];
    threshold2 = new double[L3];
    output1 = new double[L2];
    output2 = new double[L3];
    gradient1 = new double[L2];
    gradient2 = new double[L3];
    
    // Randomly initialize all connection weights 
    // and thresholds in the network within the range(0,1)
    srand((int)time(0) + rand());
    for (int h = 0; h < L2; h++) {
        for (int i = 0; i < L1; i++) {
            weight1[i][h] = rand()%1000 * 0.001 - 0.5;
        }
        threshold1[h] = rand()%1000 * 0.001 - 0.5;
    }
    for (int j = 0; j < L3; j++) {
        for (int h = 0; h < L2; h++) {
            weight2[h][j] = rand()%1000 * 0.001 - 0.5;
        }
        threshold2[j] = rand()%1000 * 0.001 - 0.5;
    }
}

BPNN::BPNN(BPNN &another)
    : L1(another.L1), L2(another.L2), L3(another.L3), lr(another.lr)
{
    weight1 = new double*[L1];
    for (int i = 0; i < L1; i++)
        weight1[i] = new double[L2];

    weight2 = new double*[L2];
    for (int i = 0; i < L2; i++)
        weight2[i] = new double[L3];

    threshold1 = new double[L2];
    threshold2 = new double[L3];
    output1 = new double[L2];
    output2 = new double[L3];
    gradient1 = new double[L2];
    gradient2 = new double[L3];

    for (int i = 0; i < L1; i++)
        for (int j = 0; j < L2; j++)
            weight1[i][j] = another.weight1[i][j];
    for (int i = 0; i < L2; i++)
        for (int j = 0; j < L3; j++)
            weight2[i][j] = another.weight2[i][j];
    for (int i = 0; i < L2; i++)
        threshold1[i] = another.threshold1[i];
    for (int i = 0; i < L3; i++)
        threshold2[i] = another.threshold2[i];
}

BPNN::~BPNN()
{
    for (int i = 0; i < L1; i++)
        delete[] weight1[i];

    for (int i = 0; i < L2; i++)
        delete[] weight2[i];

    delete[] threshold1;
    delete[] threshold2;
    delete[] output1;
    delete[] output2;
    delete[] gradient1;
    delete[] gradient2;
}

void BPNN::calculate_output(const std::vector<int> &input)
{
    for (int h = 0; h < L2; h++) {
        double sigma = 0;
        for (int i = 0; i < L1; i++) {
            sigma += (input[i] * weight1[i][h]);
        }
        output1[h] = sigmoid(sigma - threshold1[h]);
    }
   
    for (int j = 0; j < L3; j++) {
        double sigma = 0;
        for (int h = 0; h < L2; h++) {
            sigma += output1[h] * weight2[h][j];
        }
        output2[j] = sigmoid(sigma - threshold2[j]);
    }
}

void BPNN::calculate_gradient(const std::vector<int> &target)
{
    for (int j = 0; j < L3; j++) {
        gradient2[j] = output2[j] * (1.0 - output2[j]) * (target[j] - output2[j]);
    }

    for (int h = 0; h < L2; h++) {
        double sigma = 0.0;
        for (int j = 0; j < L3; j++) {
            sigma += weight2[h][j] * gradient2[j];
        }
        gradient1[h] = output1[h] * (1.0 - output1[h]) * sigma;
    }
}

void BPNN::update_weight(const std::vector<int> &input)
{
    for (int j = 0; j < L3; j++) {
        for (int h = 0; h < L2; h++) {
            weight2[h][j] += lr * gradient2[j] * output1[h];
        }
        threshold1[j] -= lr * gradient2[j];
    }
    for (int h = 0; h < L2; h++) {
        for (int i = 0; i < L1; i++) {
            weight1[i][h] += lr * gradient1[h] * input[i];
        }
        threshold2[h] -= lr * gradient1[h];
    }
}

void BPNN::train(const std::vector<int> &input, const std::vector<int> &target)
{
    calculate_output(input);
    calculate_gradient(target);
    update_weight(input);
}

std::vector<double> BPNN::test(const std::vector<int> &input)
{
    calculate_output(input);
    vector<double> res(output2, output2+L3);
    return res;
}
