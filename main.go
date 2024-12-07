package main

import (
	"fmt"
	"nn/datasets"
	"nn/network/activation"
	"nn/network/layer"
	"nn/network/loss"
	"nn/network/models"
	weightinitialization "nn/network/weight_initialization"
	"nn/trainer"
)

func main() {
	train := false

	if train {
		trainer.Train()
	}

	layer1 := layer.NewLayer()
	layer2 := layer.NewLayer()

	layer1.Init(
		9,                                     // Number of input size
		18,                                    // Number of neuron output
		activation.Relu,                       // Activation Function
		loss.BinaryCrossEntropyDerivative,     // Loss Function Derivative
		activation.DerivativeRelu,             // Activation Function Derivative
		weightinitialization.HeInitialization, // Weight Initialization
	)
	layer2.Init(
		18,
		1,
		activation.Sigmoid,
		loss.BinaryCrossEntropyDerivative,
		activation.DerivativeSigmoid,
		weightinitialization.XavierInitialization,
	)

	x := datasets.GolfXFeature()
	y := datasets.GolfYFeature()

	layers := []layer.DenseLayer{*layer1, *layer2}

	models.ImportModel(&layers, "./models/golf_model.json")

	for i := range len(x) {
		outputLayer1 := layers[0].Forward(x[i])
		outputLayer2 := layers[1].Forward(outputLayer1)

		fmt.Printf("Actual : %f, predict : %f \n", y[i], outputLayer2)
	}
}
