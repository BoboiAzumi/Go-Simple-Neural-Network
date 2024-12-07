package weightinitialization

import (
	"math"
	"math/rand/v2"
)

func XavierInitialization(inputSize int, neuronSize int) [][]float64 {
	stdDev := math.Sqrt(2.0 / float64(inputSize+neuronSize))

	weights := make([][]float64, neuronSize)
	for i := range weights {
		weights[i] = make([]float64, inputSize)
		for j := range weights[i] {
			weights[i][j] = rand.NormFloat64() * stdDev
		}
	}

	return weights
}
