package layer

import (
	"math"
	"nn/network/models/model_struct"
	weightinitialization "nn/network/weight_initialization"
)

func softmax(values []float64) []float64 {
	maxValue := max(values)
	exp := make([]float64, len(values))
	sumExp := 0.0

	for i, v := range values {
		exp[i] = math.Exp(v - maxValue)
		sumExp += exp[i]
	}

	softmax := make([]float64, len(values))
	for i := range exp {
		softmax[i] = exp[i] / sumExp
	}

	return softmax
}

func max(values []float64) float64 {
	maxVal := values[0]
	for _, v := range values {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

type SoftmaxLayer struct {
	bias           []float64
	weight         [][]float64
	neuronSize     int
	inputSize      int
	input          []float64
	output         []float64
	initialization weightinitialization.Initialization
}

func (thisLayer *SoftmaxLayer) Init(inputSize int, neuronSize int, initialization weightinitialization.Initialization) {
	thisLayer.bias = make([]float64, neuronSize)
	thisLayer.neuronSize = neuronSize
	thisLayer.inputSize = inputSize
	thisLayer.weight = initialization(inputSize, neuronSize)
}

func (thisLayer *SoftmaxLayer) Forward(x []float64) []float64 {
	thisLayer.input = x
	var logits []float64

	for i := 0; i < thisLayer.neuronSize; i++ {
		sum := 0.0
		for j := 0; j < thisLayer.inputSize; j++ {
			sum += thisLayer.weight[i][j] * x[j]
		}
		sum += thisLayer.bias[i]
		logits = append(logits, sum)
	}

	thisLayer.output = softmax(logits)
	return thisLayer.output
}

func (thisLayer *SoftmaxLayer) Backward(y []float64, learningRate float64, isHidden bool) []float64 {
	gradient := make([]float64, thisLayer.neuronSize)
	for i := 0; i < thisLayer.neuronSize; i++ {
		gradient[i] = thisLayer.output[i] - y[i]
	}

	for i := 0; i < thisLayer.neuronSize; i++ {
		for j := 0; j < thisLayer.inputSize; j++ {
			thisLayer.weight[i][j] -= learningRate * gradient[i] * thisLayer.input[j]
		}
		thisLayer.bias[i] -= learningRate * gradient[i]
	}

	prevLayerGradient := make([]float64, thisLayer.inputSize)
	for j := 0; j < thisLayer.inputSize; j++ {
		for i := 0; i < thisLayer.neuronSize; i++ {
			prevLayerGradient[j] += gradient[i] * thisLayer.weight[i][j]
		}
	}

	return prevLayerGradient
}

func (thisLayer *SoftmaxLayer) GetLayerInformation() *model_struct.LayerInformation {
	return &model_struct.LayerInformation{
		Weights:    thisLayer.weight,
		Bias:       thisLayer.bias,
		NeuronSize: thisLayer.neuronSize,
		InputSize:  thisLayer.inputSize,
	}
}

func (thisLayer *SoftmaxLayer) AssignValue(layerInformation *model_struct.LayerInformation) {
	if layerInformation.InputSize != thisLayer.inputSize || layerInformation.NeuronSize != thisLayer.neuronSize {
		panic("Cannot Assign Value into layers !")
	}

	thisLayer.weight = layerInformation.Weights
	thisLayer.bias = layerInformation.Bias
}

func NewSoftmaxLayer() *SoftmaxLayer {
	return &SoftmaxLayer{}
}
