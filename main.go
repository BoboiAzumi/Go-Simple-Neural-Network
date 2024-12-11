package main

import (
	"fmt"
	"nn/datasets"
	"nn/network/sequential"
	"nn/trainer"
)

func main() {
	train := false // True for train, false for load model

	if train {
		trainer.GolfSoftmaxTrainer()
	} else {
		Model := sequential.NewSequentialModel()
		Model.Init(9, "categoricalcrossentropy", "adam", []float64{1e-3, 0.9, 0.999, 1e-8})

		Model.AddLayer("relu", 27)
		Model.AddLayer("relu", 18)
		Model.AddLayer("relu", 9)
		Model.AddLayer("softmax", 2)

		Model.Import("./models/golf_decision_model.json")

		xTest := datasets.GolfXTestSoftmax()
		yTest := datasets.GolfYTestSoftmax()

		for i := range len(datasets.GolfXTestSoftmax()) {
			output := Model.Predict(xTest[i])
			fmt.Printf("Actual : %f, Predict : %f \n", yTest[i], output)
		}
	}
}
