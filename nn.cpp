#include <iostream>
#include <math.h>
#include <ctime>
#include <vector>
#include <algorithm>

float dotProduct(std::vector<float> vectorOne, std::vector<float> vectorTwo) {
    float returnValue = 0;
    for(int inc = 0; inc < vectorOne.size(); inc++) {
        returnValue += vectorOne[inc] * vectorTwo[inc];
    }
    return returnValue;
}
float activation(float inputValue) {
    return tanh(inputValue);
}
float activationDerivative(float inputValue) {
    float t = tanh(inputValue);
    return 1 - (t * t);
}
float lossFunction(std::vector<float> modelOutput, std::vector<float> trainingSignal) { // currently using MSE
    float total = 0;
    for(int vi = 0; vi < modelOutput.size(); vi++) {
        total += pow(trainingSignal[vi] - modelOutput[vi], 2);
    }
    return total / modelOutput.size();
}
class Neuron {
public:
    std::vector<float> weights;
    float activatedOutput;
    float biasValue;
    Neuron(int prevLayerNeuronCount) {
        this->biasValue = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //add bias weight as [0]
        for(int wi = 0; wi < prevLayerNeuronCount; wi++) {
            weights.push_back(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
        }
    }
};
class Layer {
public:
    int neuronCount;
    std::vector<Neuron> neurons;
    Layer(int inputNeuronCount, int prevLayerNeuronCount) {
        neuronCount = inputNeuronCount;
        for(int ni = 0; ni < neuronCount; ni++) {
            neurons.push_back(static_cast <Neuron> (Neuron(prevLayerNeuronCount)));
        }
    }
};
class Model {
public:
    float learningRate = 0.1;
    std::vector<Layer> layers; //index position=layer, index value=neuron count
    Model(std::vector<int> layerDescs, int inputVectorCount) {
        for(int li = 0; li < layerDescs.size(); li++) {
            int prevLayerNeurons = inputVectorCount;
            if(li > 0) {
                prevLayerNeurons = layerDescs[li - 1];
            }
            layers.push_back(static_cast <Layer> (Layer(layerDescs[li], prevLayerNeurons)));
        }
    }
    std::vector<float> forwardPropogate(std::vector<float> inputData, std::vector<float> outputSignal) {
        std::vector<float> previousLayerOutput = inputData;
        for(int layerInc = 0; layerInc < this->layers.size(); layerInc++) {
            std::vector<float> currentLayerOutput(this->layers[layerInc].neuronCount);
            for(int neuronInc = 0; neuronInc < this->layers[layerInc].neuronCount; neuronInc++) {
                float prevDotTimesWeights = dotProduct(this->layers[layerInc].neurons[neuronInc].weights, previousLayerOutput) + this->layers[layerInc].neurons[neuronInc].biasValue;
                this->layers[layerInc].neurons[neuronInc].activatedOutput = activation(prevDotTimesWeights);
                currentLayerOutput.push_back(activation(prevDotTimesWeights));
            }
            previousLayerOutput = currentLayerOutput;
            currentLayerOutput.clear();
        }
        return previousLayerOutput;
    }
    void backPropogate(std::vector<float> inputData, std::vector<float> modelOutput, std::vector<float> trainingSignal) {
        std::vector<float> errorsAtNextLayer; //used to keep track of the layer's current averaged errors for each index of weights (weights.count == prevLayer.neuronCount)
        std::vector<float> errorsAtThisLayer; //set equal to previous layer's errorsAtNextLayer for error reference while errorsAtNextLayer is used to calculate this layer's errors
        std::vector<float> activatedOutputDerivatives; //populate with derivative of current layer's neuron's weights' slopes - used to incrimentally move closer to local maxima
        
        float outputError = lossFunction(modelOutput, trainingSignal); //used to keep track of the error for each neuron - initiall this value is only used on the output layer before being set equal to a neuron's corresponding index in errorsAtThisLayer
        float delta;
        float prevLayerMatchingOutput;
        
        for(int backInc = (int) this->layers.size(); backInc > 0; backInc--) { //back propogate over layers of model
            if(backInc < (int) this->layers.size()) {
                errorsAtThisLayer.clear(); //prep vector to hold the errors of last layers' neurons' weights
                for(int vi = 0; vi < errorsAtThisLayer.size(); vi++) {
                    errorsAtThisLayer.push_back(errorsAtNextLayer[vi]);
                }
            }
            for(int oi = 0; oi < this->layers[backInc].neuronCount; oi++) {//iterate over each neuron in this layer
                if(backInc < this->layers.size()) { //if layer is not last, get error cooresponding to last layer's neurons' weights index
                    outputError = errorsAtThisLayer[oi];
                }
                delta = activationDerivative(this->layers[backInc].neurons[oi].activatedOutput) * outputError; //calculate how much current neuron should be changed for a less error-prone ouput
                for(int wi = 0; wi < this->layers[backInc].neurons[oi].weights.size(); wi++) { //iterate over the weights of current neuron
                    if(oi == 0) { //store the error of each weight
                        errorsAtNextLayer.push_back(this->layers[backInc].neurons[oi].weights[wi] * delta);
                    } else {
                        errorsAtNextLayer[wi] += this->layers[backInc].neurons[oi].weights[wi] * delta;
                    }
                    if(backInc == 1) { //get the prev layer's activated output - unless the previous layer is the input data
                        prevLayerMatchingOutput = inputData[wi];
                    } else {
                        prevLayerMatchingOutput = this->layers[backInc - 1].neurons[wi].activatedOutput;
                    }
                    this->layers[backInc].neurons[oi].weights[wi] += (prevLayerMatchingOutput * delta) * learningRate; //update the weight value to become more like the desired output of the previous layer
                }
                this->layers[backInc].neurons[oi].biasValue = (this->layers[backInc].neurons[oi].biasValue * delta) * learningRate; //update bias value of the current neuron
            }
            for(int ei = 0; ei < errorsAtNextLayer.size(); ei++) {
                errorsAtNextLayer[ei] = errorsAtNextLayer[ei] / this->layers[backInc].neuronCount;
            }
        }
    }
};

int main(int argc, const char * argv[]) {
    srand(static_cast <unsigned> (time(0)));
    return 0;
}
