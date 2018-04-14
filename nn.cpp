#include <iostream>
#include <math.h>
#include <ctime>
#include <vector>

class Neuron {
public:
    float *weightsPtr;
    int weightsCount;
    
    Neuron(int prevLayerNeuronCount) {
        this->weightsCount = prevLayerNeuronCount;
        weightsPtr = (float *) malloc(sizeof(float));
        for(int wi = 0; wi < prevLayerNeuronCount; wi++) {
            weightsPtr[wi] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }
};

class Layer {
public:
    int neuronCount;
    Neuron *neuronPtr;
    std::vector<Neuron> neurons;
    
    Layer(int neuronCount, int prevLayerNeuronCount) {
        for(int ni = 0; ni < neuronCount; ni++) {
            neuronPtr = new Neuron(90);
            neuronPtr++;
            neurons.push_back(static_cast <Neuron> (Neuron(5)));
        }
    }
};

int main(int argc, const char * argv[]) {
    srand(static_cast <unsigned> (time(0)));
    //std::cout << "Hello! I am Nick Pitoniak\n";
    Layer fuck = Layer(5, 3);
    fuck.neuronPtr--;
    std::cout << fuck.neuronPtr->weightsCount;
    //std::cout << fuck.fuck->weightsCount;
    return 0;
}
