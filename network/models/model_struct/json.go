package model_struct

type LayerInformation struct {
	Weights    [][]float64 `json:"weights"`
	Bias       []float64   `json:"bias"`
	NeuronSize int         `json:"neuron_size"`
	InputSize  int         `json:"input_size"`
}

type LayersList struct {
	Layers []LayerInformation `json:"layers"`
}
