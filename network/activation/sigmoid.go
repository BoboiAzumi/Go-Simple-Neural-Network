package activation

import "math"

func Sigmoid(x float64) float64 {
	return (1 / (1 + math.Pow(math.E, -x)))
}

func DerivativeSigmoid(y float64) float64 {
	return (y * (1 - y))
}
