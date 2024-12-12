package layer

import (
	weightinitialization "github.com/BoboiAzumi/Go-Simple-Neural-Network/network/weight_initialization"

	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/activation"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/loss"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/models/model_struct"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/optimizer"
)

type SoftmaxLayer struct {
	bias           []float64
	weight         [][]float64
	neuronSize     int
	inputSize      int
	input          []float64
	output         []float64
	initialization weightinitialization.Initialization
	opt            optimizer.Optimizer
	m              [][]float64
	v              [][]float64
	mBias          []float64
	vBias          []float64
}

func (thisLayer *SoftmaxLayer) Init(inputSize int, neuronSize int, activation activation.ActivationFunc, derivativeLoss loss.DerivativeLossFunc, derivativeActivation activation.DerivativeActivationFunc, initialization weightinitialization.Initialization) {
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

	thisLayer.output = activation.Softmax(logits)
	return thisLayer.output
}

func (thisLayer *SoftmaxLayer) SetOptimizer(opt optimizer.Optimizer) {
	thisLayer.opt = opt
}

func (thisLayer *SoftmaxLayer) Backward(y []float64, isHidden bool) []float64 {
	gradient := make([]float64, thisLayer.neuronSize)
	for i := 0; i < thisLayer.neuronSize; i++ {
		gradient[i] = thisLayer.output[i] - y[i]
	}

	if thisLayer.opt.Info() == "adam" {
		if thisLayer.m == nil || thisLayer.v == nil {
			thisLayer.m = make([][]float64, thisLayer.neuronSize)
			thisLayer.v = make([][]float64, thisLayer.neuronSize)
			for i := range thisLayer.m {
				thisLayer.m[i] = make([]float64, thisLayer.inputSize)
				thisLayer.v[i] = make([]float64, thisLayer.inputSize)
			}
		}
		if thisLayer.mBias == nil || thisLayer.vBias == nil {
			thisLayer.mBias = make([]float64, thisLayer.neuronSize)
			thisLayer.vBias = make([]float64, thisLayer.neuronSize)
		}
	}

	prevLayerGradient := make([]float64, thisLayer.inputSize)
	for j := 0; j < thisLayer.inputSize; j++ {
		for i := 0; i < thisLayer.neuronSize; i++ {
			prevLayerGradient[j] += gradient[i] * thisLayer.weight[i][j]
		}
	}

	for i := 0; i < thisLayer.neuronSize; i++ {
		for j := 0; j < thisLayer.inputSize; j++ {
			thisLayer.weight[i][j] -= thisLayer.opt.UpdateWeight(gradient[i]*thisLayer.input[j], &thisLayer.m, &thisLayer.v, i, j)
		}
		thisLayer.bias[i] -= thisLayer.opt.UpdateBias(gradient[i], &thisLayer.mBias, &thisLayer.vBias, i)
	}

	if thisLayer.opt.Info() == "adam" {
		thisLayer.opt.Step()
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
