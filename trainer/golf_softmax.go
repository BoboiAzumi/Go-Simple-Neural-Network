package trainer

import (
	"fmt"
	"nn/datasets"
	"nn/network/loss"
	"nn/network/sequential"
)

func GolfSoftmaxTrainer() {
	x := datasets.GolfXFeatureSoftmax()
	y := datasets.GolfYFeatureSoftmax()

	Model := sequential.NewSequentialModel()
	Model.Init(9, "categoricalcrossentropy", "adam", []float64{1e-3, 0.9, 0.999, 1e-8})

	Model.AddLayer("relu", 27)
	Model.AddLayer("relu", 18)
	Model.AddLayer("relu", 9)
	Model.AddLayer("softmax", 2)

	stopLoss := 0.000001
	backprop := true
	// Training
	for epoch := range 4000 {
		for batch := range len(x) {
			if !backprop {
				continue
			}
			Model.Predict(x[batch])
			Model.Backward(y[batch])
		}

		loss_val := 0.0

		for i := range len(x) {
			output := Model.Predict(x[i])
			loss_val += loss.CategoricalCrossEntropyLoss(output, y[i])
		}

		loss_val = loss_val / float64(len(x))
		if loss_val < stopLoss {
			backprop = false
		}

		if epoch%100 == 0 {
			fmt.Printf("Loss : %f \n", loss_val)
		}
	}

	xTest := datasets.GolfXTestSoftmax()
	yTest := datasets.GolfYTestSoftmax()

	for i := range len(datasets.GolfXTestSoftmax()) {
		output := Model.Predict(xTest[i])
		fmt.Printf("Actual : %f, Predict : %f \n", yTest[i], output)
	}

	Model.Export("./models/golf_decision_model.json")
}
