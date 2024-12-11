package sequential

import (
	"nn/network/layer"
	"nn/network/models"
)

type SequentialModel struct {
	layer      []layer.Layer
	inputShape int
	loss       string
	opt        string
	optParam   []float64
}

func (sm *SequentialModel) Init(inputShape int, lossFunc string, opt string, optParam []float64) {
	sm.inputShape = inputShape
	sm.layer = []layer.Layer{}
	sm.loss = lossFunc
	sm.opt = opt
	sm.optParam = optParam
}

func (sm *SequentialModel) AddLayer(activation string, size int) {
	inputShape := 0

	if len(sm.layer) == 0 {
		inputShape = sm.inputShape
	} else {
		inputShape = sm.layer[len(sm.layer)-1].GetLayerInformation().NeuronSize
	}

	layerType := ChooseLayer(activation)

	layerType.Init(
		inputShape,
		size,
		ChooseActivation(activation),
		ChooseDerivativeLoss(sm.loss),
		ChooseDerivativeActivation(activation),
		ChooseInitialization(activation),
	)

	layerType.SetOptimizer(ChooseOptimizer(sm.opt, sm.optParam))

	sm.layer = append(
		sm.layer,
		layerType,
	)
}

func (sm *SequentialModel) Predict(x []float64) []float64 {
	var y []float64

	for i := range len(sm.layer) {
		var input []float64

		if i == 0 {
			input = x
		} else {
			input = y
		}

		y = sm.layer[i].Forward(input)
	}

	return y
}

func (sm *SequentialModel) Backward(y []float64) {
	var gradientLoss []float64

	for i := range len(sm.layer) {
		index := len(sm.layer) - 1

		if i == 0 {
			gradientLoss = sm.layer[index-i].Backward(y, false)
		} else {
			gradientLoss = sm.layer[index-i].Backward(gradientLoss, true)
		}
	}
}

func (sm *SequentialModel) Export(path string) {
	models.Export(sm.layer, path)
}

func (sm *SequentialModel) Import(path string) {
	models.ImportModel(&sm.layer, path)
}

func NewSequentialModel() *SequentialModel {
	return &SequentialModel{}
}
