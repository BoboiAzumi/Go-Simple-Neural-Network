package activation

import "math"

func Softmax(values []float64) []float64 {
	maxValue := max(values)
	exp := make([]float64, len(values))
	sumExp := 0.0

	for i, v := range values {
		exp[i] = math.Exp(v - maxValue)
		sumExp += exp[i]
	}

	softmax := make([]float64, len(values))
	for i := range exp {
		softmax[i] = exp[i] / sumExp
	}

	return softmax
}

func max(values []float64) float64 {
	maxVal := values[0]
	for _, v := range values {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}
