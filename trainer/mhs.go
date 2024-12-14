package trainer

import (
	"fmt"

	"github.com/BoboiAzumi/Go-Simple-Neural-Network/csv"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/loss"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/sequential"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/utils"
)

func TrainMHS() {
	loadedCSV := csv.Load("./csv/resources/klasifikasimhs.csv")
	ohe := utils.NewOneHotEncoder()
	ohe.ScanUnique(loadedCSV.Data, 1)

	Scaler1 := utils.NewMinMaxScaler()
	Scaler1.Fit(loadedCSV.Data, 2)
	Scaler2 := utils.NewMinMaxScaler()
	Scaler2.Fit(loadedCSV.Data, 3)
	Scaler3 := utils.NewMinMaxScaler()
	Scaler3.Fit(loadedCSV.Data, 4)

	tempatTinggal := utils.StringToFloat(loadedCSV.Data, 0)
	pekerjaanOrangTua := ohe.Encoding(loadedCSV.Data, 1)
	penghasilanOrangTua := Scaler1.Transform(loadedCSV.Data, 2)
	jumlahTanggungan := Scaler2.Transform(loadedCSV.Data, 3)
	kendaraan := Scaler3.Transform(loadedCSV.Data, 4)
	kelayakan := utils.StringToFloat(loadedCSV.Data, 5)

	utils.Concat(&tempatTinggal, &pekerjaanOrangTua)
	utils.Concat(&tempatTinggal, &penghasilanOrangTua)
	utils.Concat(&tempatTinggal, &jumlahTanggungan)
	utils.Concat(&tempatTinggal, &kendaraan)

	Model := sequential.NewSequentialModel()
	Model.Init(12, "binarycrossentropy", "adam", []float64{1e-4, 0.9, 0.999, 1e-8})
	Model.AddLayer("relu", 128)
	Model.AddLayer("sigmoid", 1)

	epochs := 1000

	for epoch := range epochs {
		for j := range len(loadedCSV.Data) {
			Model.Predict(tempatTinggal[j])
			Model.Backward(kelayakan[j])
		}

		loss_val := 0.0

		for j := range len(loadedCSV.Data) {
			output := Model.Predict(tempatTinggal[j])
			loss_val += loss.BinaryCrossEntropy(output, kelayakan[j])
		}

		if epoch%100 == 0 {
			loss_val = loss_val / float64(len(loadedCSV.Data))
			fmt.Printf("%d / %d, Loss : %f\n", epoch, epochs, loss_val)
		}
	}

	for j := range len(loadedCSV.Data) {
		output := Model.Predict(tempatTinggal[j])
		fmt.Printf("%d. Actual : %f, Predict : %f\n", j, kelayakan[j], output)
	}

	Model.Export("./models/klasifikasimhs.json")
}
