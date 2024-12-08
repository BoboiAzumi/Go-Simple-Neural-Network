package loss

import "math"

func CategoricalCrossEntropyLoss(yPred, yTrue []float64) float64 {
	if len(yTrue) != len(yPred) {
		panic("Target and prediction vectors must have the same length")
	}

	loss := 0.0
	for i := range yTrue {
		if yTrue[i] == 1 {
			// Tambahkan stabilitas numerik untuk menghindari log(0)
			epsilon := 1e-15
			pred := math.Max(yPred[i], epsilon) // Stabilitas numerik
			loss -= math.Log(pred)
		}
	}
	return loss
}
