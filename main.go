package main

import (
	"fmt"
	"runtime"

	"github.com/BoboiAzumi/Go-Simple-Neural-Network/csv"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/loss"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/sequential"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/utils"
)

func main() {
	numCPU := runtime.NumCPU()
	runtime.GOMAXPROCS(numCPU)

	loadedCSV := csv.Load("./csv/resources/Iris.csv")
	Scaler1 := utils.NewMinMaxScaler()
	Scaler1.Fit(loadedCSV.Data, 1)
	Scaler2 := utils.NewMinMaxScaler()
	Scaler2.Fit(loadedCSV.Data, 2)
	Scaler3 := utils.NewMinMaxScaler()
	Scaler3.Fit(loadedCSV.Data, 3)
	Scaler4 := utils.NewMinMaxScaler()
	Scaler4.Fit(loadedCSV.Data, 4)

	ohe := utils.NewOneHotEncoder()
	ohe.ScanUnique(loadedCSV.Data, 5)
	Y := ohe.Encoding(loadedCSV.Data, 5)

	X := Scaler1.Transform(loadedCSV.Data, 1)
	X1 := Scaler2.Transform(loadedCSV.Data, 2)
	X2 := Scaler3.Transform(loadedCSV.Data, 3)
	X3 := Scaler4.Transform(loadedCSV.Data, 4)
	utils.Concat(&X, &X1)
	utils.Concat(&X, &X2)
	utils.Concat(&X, &X3)

	Model := sequential.NewSequentialModel()
	Model.Init(4, "categoricalcrossentropy", "adam", []float64{1e-4, 0.9, 0.999, 1e-8})
	Model.AddLayer("relu", 64)
	Model.AddLayer("relu", 32)
	Model.AddLayer("softmax", 3)

	epochs := 1000

	for epoch := range epochs {
		for j := range len(loadedCSV.Data) {
			Model.Predict(X[j])
			Model.Backward(Y[j])
		}

		loss_val := 0.0

		for j := range len(loadedCSV.Data) {
			output := Model.Predict(X[j])
			loss_val += loss.CategoricalCrossEntropyLoss(output, Y[j])
		}

		if epoch%100 == 0 {
			loss_val = loss_val / float64(len(loadedCSV.Data))
			fmt.Printf("%d / %d, Loss : %f\n", epoch, epochs, loss_val)
		}
	}

	for j := range len(loadedCSV.Data) {
		output := Model.Predict(X[j])
		fmt.Printf("%d. Actual : %f, Predict : %f\n", j, Y[j], output)
	}

	Model.Export("./models/iris.json")
}
