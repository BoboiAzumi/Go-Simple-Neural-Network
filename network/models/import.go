package models

import (
	"encoding/json"
	"io/ioutil"
	"nn/network/layer"
	"nn/network/models/model_struct"
	"os"
)

func ImportModel(layers *[]layer.Layer, path string) {
	file, err := os.Open(path)

	if err != nil {
		panic("File not found")
	}
	defer file.Close()

	data, err := ioutil.ReadAll(file)

	if err != nil {
		panic("Error read file")
	}

	var layerList model_struct.LayersList
	err = json.Unmarshal(data, &layerList)

	if err != nil {
		panic("Error when convert JSON")
	}

	if len(*layers) != len(layerList.Layers) {
		panic("Model load error")
	}

	for i := range len(*layers) {
		(*layers)[i].AssignValue(&layerList.Layers[i])
	}
}
