package layer

import (
	"nn/network/activation"
	"nn/network/loss"
	"nn/network/models/model_struct"
	"nn/network/optimizer"
	weightinitialization "nn/network/weight_initialization"
)

type Layer interface {
	Init(inputSize int,
		neuronSize int,
		activation activation.ActivationFunc,
		derivativeLoss loss.DerivativeLossFunc,
		derivativeActivation activation.DerivativeActivationFunc,
		initialization weightinitialization.Initialization,
	)
	Forward(x []float64) []float64
	SetOptimizer(opt optimizer.Optimizer)
	Backward(y []float64, isHidden bool) []float64
	GetLayerInformation() *model_struct.LayerInformation
	AssignValue(layerInformation *model_struct.LayerInformation)
}
