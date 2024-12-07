package weightinitialization

import "math/rand/v2"

func RandomInitialization(inputSize int, neuronSize int) [][]float64 {
	weights := make([][]float64, neuronSize)
	for i := range neuronSize {
		weights[i] = make([]float64, inputSize)
		for j := range inputSize {
			weights[i][j] = rand.Float64()
		}
	}

	return weights
}
