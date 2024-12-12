package loss

import "math"

func MSEError(myout []float64, ayout []float64) float64 {
	if len(myout) != len(ayout) {
		panic("Actual Y and Predict Y length not same")
	}
	sum := 0.0

	for i := range len(myout) {
		sum += math.Pow((ayout[i] - myout[i]), 2.0)
	}

	return (1.0 / float64(len(myout)) * sum)
}

func MSEDerivative(myout float64, ayout float64, sample int) float64 {
	return ((-2 / float64(sample)) * (ayout - myout))
}
