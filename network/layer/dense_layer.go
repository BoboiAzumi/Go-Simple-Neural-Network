package layer

import (
	"sync"

	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/activation"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/loss"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/models/model_struct"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/optimizer"
	weightinitialization "github.com/BoboiAzumi/Go-Simple-Neural-Network/network/weight_initialization"
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
	opt                          optimizer.Optimizer
	m                            [][]float64
	v                            [][]float64
	mBias                        []float64
	vBias                        []float64
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

func (thisLayer *DenseLayer) SetOptimizer(opt optimizer.Optimizer) {
	thisLayer.opt = opt
}

func (thisLayer *DenseLayer) Backward(y []float64, isHidden bool) []float64 {
	var loss_derivative []float64 = make([]float64, thisLayer.neuronSize)
	var activation_derivative []float64 = make([]float64, thisLayer.neuronSize)
	var wg sync.WaitGroup
	if isHidden {
		loss_derivative = y
	} else {
		for i := range thisLayer.neuronSize {
			wg.Add(1)
			go func() {
				defer wg.Done()
				loss_derivative[i] = thisLayer.derivativeLossFunction(thisLayer.output[i], y[i], thisLayer.neuronSize)
			}()
		}
		wg.Wait()
	}

	for i := range thisLayer.neuronSize {
		wg.Add(1)
		go func() {
			defer wg.Done()
			activation_derivative[i] = thisLayer.derivativeActivationFunction(thisLayer.output[i])
		}()
	}
	wg.Wait()

	prevGradient := make([]float64, thisLayer.inputSize)

	for i := range thisLayer.inputSize {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sumLossDerivative := 0.0
			sumActivationDerivative := 0.0
			sumInput := 0.0
			sumWeight := 0.0
			for j := range thisLayer.neuronSize {
				sumLossDerivative += loss_derivative[j]
				sumActivationDerivative += activation_derivative[j]
				sumInput += thisLayer.input[i]
				sumWeight += thisLayer.weight[j][i]
			}

			prevGradient[i] = sumLossDerivative * sumActivationDerivative * sumInput * sumWeight
		}()
	}

	wg.Wait()

	if thisLayer.opt.Info() == "adam" {
		if thisLayer.m == nil || thisLayer.v == nil {
			thisLayer.m = make([][]float64, thisLayer.neuronSize)
			thisLayer.v = make([][]float64, thisLayer.neuronSize)
			for i := range thisLayer.m {
				wg.Add(1)
				go func() {
					defer wg.Done()
					thisLayer.m[i] = make([]float64, thisLayer.inputSize)
					thisLayer.v[i] = make([]float64, thisLayer.inputSize)
				}()
			}
			wg.Wait()
		}
		if thisLayer.mBias == nil || thisLayer.vBias == nil {
			thisLayer.mBias = make([]float64, thisLayer.neuronSize)
			thisLayer.vBias = make([]float64, thisLayer.neuronSize)
		}
	}

	for i := range thisLayer.neuronSize {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range thisLayer.inputSize {
				thisLayer.weight[i][j] -= thisLayer.opt.UpdateWeight(loss_derivative[i]*activation_derivative[i]*thisLayer.input[j], &thisLayer.m, &thisLayer.v, i, j)
			}
			thisLayer.bias[i] -= thisLayer.opt.UpdateBias(loss_derivative[i]*activation_derivative[i]*1, &thisLayer.mBias, &thisLayer.vBias, i)
		}()
	}
	wg.Wait()

	if thisLayer.opt.Info() == "adam" {
		thisLayer.opt.Step()
	}

	return prevGradient
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
