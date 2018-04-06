#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct ModelDataset {
    float *modelDataPointer;
    int setCount;
    int setDimensions;
};

struct PerceptronModel {
    int wDimensions;
    int learningEpochs;
    float *weightsPointer;
    float bias;
    float biasWeight;
    float learningRate;
};

struct ModelDataset fetchDataset(int datasetCount, int inputDimensions) {
    float *dataPointerOg = (float *) malloc(((inputDimensions + 1) * sizeof(float)) * datasetCount);
    float *dataPointer = &*dataPointerOg;
    int dim1Inc = 0;
    int dim2Inc = 0;
    int currChar;
    FILE *dataFile = fopen("dataset.txt", "r");
    if (dataFile) {
        char cnarr[100];
        char *cnp = &cnarr[0];
        while ((currChar = getc(dataFile)) != EOF) {
            if((char) currChar == ',') {
                *cnp = '\0';
                cnp = &cnarr[0];
		*dataPointer++ = (float) atof(cnarr);
            } else if((char) currChar == '\n') {
                *cnp = '\0';
                cnp = &cnarr[0];
		*dataPointer++ = (float) atof(cnarr);
            } else {
                *cnp++ = (char) currChar;
            }
        }
        fclose(dataFile);
    }
    struct ModelDataset RetData;
    RetData.modelDataPointer = dataPointerOg;
    RetData.setCount = datasetCount;
    RetData.setDimensions = inputDimensions;    
    return RetData;
}

int trainModel(struct PerceptronModel *learningModel, float *data, int signal) {
    int modelOutput;
    float lazyDotProduct = learningModel->biasWeight;
    for(int wi = 0; wi < learningModel->wDimensions; wi = wi + 1) {
        lazyDotProduct = lazyDotProduct + (data[wi] * learningModel->weightsPointer[wi]);    
    }
    if(lazyDotProduct >= 0.0) {
        modelOutput = 1;
    } else {
        modelOutput = 0;
    }
    for(int wi = 0; wi < learningModel->wDimensions; wi = wi + 1) {
        learningModel->weightsPointer[wi] = learningModel->weightsPointer[wi] + learningModel->learningRate * (signal - modelOutput) * data[wi];
    }
    learningModel->biasWeight = learningModel->biasWeight + learningModel->learningRate * (signal - modelOutput);
    if(signal - modelOutput == 0) {
        return 1;
    } else {
        return 0;
    }
}

struct PerceptronModel generateModel(int dataDimensions, int learningEpochs) {
    float *wPointerOg = (float*) malloc(dataDimensions * sizeof(float));
    float *wPointer = &*wPointerOg;
    for(int wi = 0; wi < dataDimensions; wi = wi + 1){
        srand(((unsigned int)time(NULL)) * (wi + 1));
        *wPointer++ = (float) rand() / RAND_MAX;
    }
    float biasWeight = (float) rand() / RAND_MAX;
    struct PerceptronModel LearningModel;
    LearningModel.bias = 1.0;
    LearningModel.biasWeight = biasWeight;
    LearningModel.learningEpochs = learningEpochs;
    LearningModel.learningRate = 0.1;
    LearningModel.wDimensions = dataDimensions;
    LearningModel.weightsPointer = wPointerOg;
    return LearningModel;
}

int main(int argc, const char * argv[]) {
    int datasetRows = 16;
    int datasetDim = 2;
    struct ModelDataset TrainingData = fetchDataset(datasetRows, datasetDim);
    struct PerceptronModel LearningModel = generateModel(datasetDim, 1000);
    for(int epoch = 0; epoch < LearningModel.learningEpochs; epoch = epoch + 1) {
        for(int setInc = 0; setInc < TrainingData.setCount; setInc = setInc + 1) {
            trainModel(&LearningModel, &TrainingData.modelDataPointer[setInc * (datasetDim + 1)], TrainingData.modelDataPointer[(setInc * (datasetDim  + 1)) + TrainingData.setDimensions]);
        }
    }
    return 0;
}

