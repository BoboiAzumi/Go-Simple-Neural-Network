package activation

import "math"

func Tanh(x float64) float64 {
	a := math.Pow(math.E, x) - math.Pow(math.E, -x)
	b := math.Pow(math.E, x) + math.Pow(math.E, -x)

	return a / b
}

func DerivativeTanh(y float64) float64 {
	return 1 - math.Pow(y, 2)
}
