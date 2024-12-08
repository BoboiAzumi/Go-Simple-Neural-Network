package trainer

import (
	"fmt"
	"math"
	"nn/datasets"
	"nn/network/activation"
	"nn/network/layer"
	"nn/network/loss"
	weightinitialization "nn/network/weight_initialization"
)

func TrainSoftmax() {
	layer1 := layer.NewLayer()
	layer2 := layer.NewSoftmaxLayer()

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
		2,
		weightinitialization.XavierInitialization,
	)

	x := datasets.GolfXFeatureSoftmax()
	y := datasets.GolfYFeatureSoftmax()
	xTest := datasets.GolfXTestSoftmax()
	yTest := datasets.GolfYTestSoftmax()

	epoch := 20000
	learningRate := 1e-4
	backprop := true
	stopLoss := 0.1 // Early Stop

	for i := range epoch {
		if !backprop {

		}
		fmt.Printf("===================\n")
		fmt.Printf("Epoch %d \n", i)
		length := len(x)
		lossTotal := 0.0
		for j := range length {
			outputLayer1 := layer1.Forward(x[j])
			outputLayer2 := layer2.Forward(outputLayer1)

			lossTotal += loss.CategoricalCrossEntropyLoss(outputLayer2, y[j])

			backLayer1 := layer2.Backward(y[j], learningRate, false)
			layer1.Backward(backLayer1, learningRate, true)
		}

		lossTotal = lossTotal / float64(length)
		if lossTotal < stopLoss {
			backprop = false
		}

		if math.IsNaN(lossTotal) {
			backprop = false
		}

		fmt.Printf("Loss : %f \n", lossTotal)
	}

	fmt.Println("\nTest")

	for i := range len(xTest) {
		outputLayer1 := layer1.Forward(xTest[i])
		outputLayer2 := layer2.Forward(outputLayer1)

		fmt.Printf("Actual : %f, predict : %f \n", yTest[i], outputLayer2)
	}

}
