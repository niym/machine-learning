#ifndef BPNN_H
#define BPNN_H

#include <vector>
#include <cmath>

class BPNN
{
public:
    BPNN(int input_length, int hidden_length, int output_length, 
            double learn_rate = 0.35);
    BPNN(BPNN &another);

    ~BPNN();

    void train(const std::vector<int> &input, 
            const std::vector<int> &target);

    std::vector<double> test(const std::vector<int> &input);


private:
    int L1;    // input layer length
    int L2;    // hidden layer length
    int L3;    // output layer length
    
    double lr;  // learning rate
    
    double** weight1;
    double** weight2;
    double* threshold1;
    double* threshold2;
    double* output1;
    double* output2;
    double* gradient1;
    double* gradient2;

    double sigmoid(const double x)
    {
        return 1.0 / (1.0 + exp(-x));
    }

    void calculate_output(const std::vector<int> &input);
    void calculate_gradient(const std::vector<int> &target);
    void update_weight(const std::vector<int> &input);
};

#endif

