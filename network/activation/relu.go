package activation

import "math"

func Relu(x float64) float64 {
	return math.Max(0, x)
}

func DerivativeRelu(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}
