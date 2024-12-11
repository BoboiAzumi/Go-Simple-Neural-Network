package loss

import "math"

func CategoricalCrossEntropyLoss(yPred, yTrue []float64) float64 {
	if len(yTrue) != len(yPred) {
		panic("Target and prediction vectors must have the same length")
	}

	loss := 0.0
	for i := range yTrue {
		if yTrue[i] == 1 {
			epsilon := 1e-15
			pred := math.Max(yPred[i], epsilon)
			loss -= math.Log(pred)
		}
	}
	return loss
}
