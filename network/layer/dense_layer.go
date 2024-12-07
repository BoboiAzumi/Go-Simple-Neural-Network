package layer

import (
	"nn/network/activation"
	"nn/network/loss"
	"nn/network/models/model_struct"
	weightinitialization "nn/network/weight_initialization"
)

type DenseLayer struct {
	bias                         []float64
	weight                       [][]float64
	neuronSize                   int
	inputSize                    int
	input                        []float64
	output                       []float64
	derivativeLossFunction       loss.DerivativeLossFunc
	activationFunction           activation.ActivationFunc
	derivativeActivationFunction activation.DerivativeActivationFunc
	initialization               weightinitialization.Initialization
}

func (thisLayer *DenseLayer) Init(inputSize int, neuronSize int, activation activation.ActivationFunc, derivativeLoss loss.DerivativeLossFunc, derivativeActivation activation.DerivativeActivationFunc, initialization weightinitialization.Initialization) {
	thisLayer.activationFunction = activation
	thisLayer.bias = []float64{}
	thisLayer.weight = [][]float64{}
	thisLayer.neuronSize = neuronSize
	thisLayer.inputSize = inputSize
	thisLayer.derivativeLossFunction = derivativeLoss
	thisLayer.derivativeActivationFunction = derivativeActivation
	thisLayer.initialization = initialization

	for range neuronSize {
		thisLayer.bias = append(thisLayer.bias, 0)
	}

	thisLayer.weight = initialization(inputSize, neuronSize)
}

func (thisLayer *DenseLayer) Forward(x []float64) []float64 {
	thisLayer.input = x
	var matrix []float64 = []float64{}
	for i := range len(thisLayer.weight) {
		var sum float64 = 0
		for j := range len(thisLayer.weight[i]) {
			sum += thisLayer.weight[i][j] * x[j]
		}
		sum += thisLayer.bias[i]
		sum = thisLayer.activationFunction(sum)
		matrix = append(matrix, sum)
	}
	thisLayer.output = matrix

	return thisLayer.output
}

func (thisLayer *DenseLayer) Backward(y []float64, learningRate float64, isHidden bool) []float64 {
	var lossOutGradient []float64 = []float64{}
	var derivativeActivation []float64 = []float64{}
	var inputWeightGradient [][]float64 = [][]float64{}

	if isHidden {
		lossOutGradient = y
	} else {
		for i := range thisLayer.neuronSize {
			temp := thisLayer.derivativeLossFunction(thisLayer.output[i], y[i])
			lossOutGradient = append(lossOutGradient, temp)
		}
	}

	for i := range thisLayer.neuronSize {
		temp := thisLayer.derivativeActivationFunction(thisLayer.output[i])
		derivativeActivation = append(derivativeActivation, temp)
	}

	for range thisLayer.neuronSize {
		inputWeightGradient = append(inputWeightGradient, thisLayer.input)
	}

	oldWeight := thisLayer.weight

	for i := range thisLayer.neuronSize {
		for j := range thisLayer.inputSize {
			lossWeight := lossOutGradient[i] * derivativeActivation[i] * inputWeightGradient[i][j]
			temp := thisLayer.weight[i][j] - (learningRate * lossWeight)
			thisLayer.weight[i][j] = temp
		}
	}

	for i := range thisLayer.neuronSize {
		lossBias := lossOutGradient[i] * derivativeActivation[i] * 1
		temp := thisLayer.bias[i] - (learningRate * lossBias)
		thisLayer.bias[i] = temp
	}

	sumLossOutGradient := 0.0
	sumDerivativeActivation := 0.0
	var sumInput []float64 = []float64{}
	var sumOldWeight []float64 = []float64{}

	for i := range thisLayer.neuronSize {
		temp := lossOutGradient[i]
		sumLossOutGradient += temp
	}

	for i := range thisLayer.neuronSize {
		temp := derivativeActivation[i]
		sumDerivativeActivation += temp
	}

	for i := range thisLayer.inputSize {
		var sumInputPartial float64 = 0.0
		for j := range thisLayer.neuronSize {
			temp := inputWeightGradient[j][i]
			sumInputPartial += temp
		}
		sumInput = append(sumInput, sumInputPartial)
	}

	for i := range thisLayer.inputSize {
		var sumOldWeightPartial float64 = 0.0
		for j := range thisLayer.neuronSize {
			temp := oldWeight[j][i]
			sumOldWeightPartial += temp
		}
		sumOldWeight = append(sumOldWeight, sumOldWeightPartial)
	}

	var lossOutForward []float64 = []float64{}

	for i := range thisLayer.inputSize {
		temp := sumLossOutGradient * sumDerivativeActivation * sumInput[i] * sumOldWeight[i]
		lossOutForward = append(lossOutForward, temp)
	}

	return lossOutForward
}

func (thisLayer *DenseLayer) GetLayerInformation() *model_struct.LayerInformation {
	return &model_struct.LayerInformation{
		Weights:    thisLayer.weight,
		Bias:       thisLayer.bias,
		NeuronSize: thisLayer.neuronSize,
		InputSize:  thisLayer.inputSize,
	}
}

func (thisLayer *DenseLayer) AssignValue(layerInformation *model_struct.LayerInformation) {
	if layerInformation.InputSize != thisLayer.inputSize || layerInformation.NeuronSize != thisLayer.neuronSize {
		panic("Cannot Assign Value into layers !")
	}

	thisLayer.weight = layerInformation.Weights
	thisLayer.bias = layerInformation.Bias
}

func NewLayer() *DenseLayer {
	return &DenseLayer{}
}
