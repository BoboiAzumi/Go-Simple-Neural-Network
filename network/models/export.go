package models

import (
	"encoding/json"
	"os"

	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/layer"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/models/model_struct"
)

func Export(layers []layer.Layer, path string) {
	var layerlist model_struct.LayersList = model_struct.LayersList{
		Layers: []model_struct.LayerInformation{},
	}

	for i := range len(layers) {
		layersInformation := layers[i].GetLayerInformation()
		layerlist.Layers = append(layerlist.Layers, *layersInformation)
	}

	jsonMarshal, err := json.Marshal(layerlist)

	if err != nil {
		panic("Error when convert weight into json")
	}

	err = os.WriteFile(path, jsonMarshal, 0777)

	if err != nil {
		panic("Error when writing weight into json file")
	}
}
