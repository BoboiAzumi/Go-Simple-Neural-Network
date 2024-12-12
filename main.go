package main

import (
	"fmt"
	"os"

	"github.com/BoboiAzumi/Go-Simple-Neural-Network/datasets"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/sequential"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/trainer"
)

func main() {
	method := os.Args[1:]

	if len(method) != 0 {
		if method[0] == "-t" {
			trainer.GolfSoftmaxTrainer()
		}
	}

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
