#include <fstream>
#include <iostream>
#include "bpnn.h"

using namespace std;

#define L1 784
#define L2 100
#define L3 10

void training(BPNN &nnet)
{
    ifstream train_images("./data/train-images.idx3-ubyte");
    ifstream train_labels("./data/train-labels.idx1-ubyte");
    
    if(!train_images || !train_labels) { 
        cout << "Cannot open file.\n"; 
        return; 
    } 
    
    train_images.seekg(16);
    train_labels.seekg(8);

    char image_buf[L1];
    char label_buf[10];
    vector<int> input(L1);
    vector<int> target(L3);

    int count = 0;
    cout << "training start ... " << endl;
    while (train_images.read(image_buf, L1) 
            && train_labels.read(label_buf, 1)) {
        for (int i = 0; i < L1; i++) {
            if ((unsigned char)image_buf[i] < 128)
                input[i] = 0;
            else
                input[i] = 1;
        }

        int target_value = (unsigned int)label_buf[0];
        for (int j = 0; j < L3; j++) {
            target[j] = 0;
        }
        target[target_value] = 1;

        nnet.train(input, target);
        
        count++;
        if (count % 1000 == 0)
            cout << "train times " << count << endl;
    }

    train_images.close();
    train_labels.close();
}


void testing(BPNN &nnet)
{
    int count = 0;
    int correct = 0;
    int mistake = 0;

    ifstream test_images("./data/t10k-images.idx3-ubyte");
    ifstream test_labels("./data/t10k-labels.idx1-ubyte");
    test_images.seekg(16);
    test_labels.seekg(8);

    char input_buf[L1];
    char label_buf[10];
    vector<int> input(L1);
    vector<int> target(L3);

    while (test_images.read(input_buf, L1) 
            && test_labels.read(label_buf, 1)) {
        for (int i = 0; i < L1; i++) {
            if ((unsigned char)input_buf[i] < 128)
                input[i] = 0;
            else
                input[i] = 1;
        }
        int target_value = (unsigned int)label_buf[0];
        for (int j = 0; j < L3; j++) {
            target[j] = 0;
        }
        target[target_value] = 1;
        
        vector<double> output = nnet.test(input);

        double max_value = -9999;
        int max_index = 0;
        for (int i = 0; i < L3; i++) {
            if (output[i] > max_value) {
                max_value = output[i];
                max_index = i;
            }
        }

        count++;
        if (target[max_index] == 1) {
            correct++;
            cout << "NO." << count << " correct" << endl;
        }
        else {
            mistake++;
            cout << "NO." << count << " mistake" << endl;
        }
    }
    cout << "\n--------------------\n";
    cout << correct << " correct, " << mistake << " mistake, accuracy " 
        << (double)correct/(double)count << endl;

    test_images.close();
    test_labels.close();
}

int main()
{
    BPNN nnet(L1, L2, L3);
    training(nnet);
    testing(nnet);
}


