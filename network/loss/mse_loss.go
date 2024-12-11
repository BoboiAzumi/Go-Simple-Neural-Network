package loss

import "math"

func MSEError(myout []float64, ayout []float64) float64 {
	if len(myout) != len(ayout) {
		panic("Actual Y and Predict Y length not same")
	}

	return math.Pow((ayout[0] - myout[0]), 2.0)
}

func MSEDerivative(myout float64, ayout float64) float64 {
	return ayout - myout
}
