package loss

import "math"

func BinaryCrossEntropy(myout []float64, ayout []float64) float64 {
	if len(myout) != len(ayout) {
		panic("Actual Y and Predict Y length not same")
	}

	epsilon := 1e-15

	var sum float64 = 0
	for i := range len(myout) {
		if myout[i] < epsilon {
			myout[i] = epsilon
		} else if myout[i] > 1-epsilon {
			myout[i] = 1 - epsilon
		}

		logMyout := math.Log(myout[i])
		logIMyout := math.Log(1 - myout[i])

		a := ayout[i] * (logMyout)
		b := (1 - ayout[i]) * logIMyout
		sum += a + b
	}

	if sum == 0 {
		sum = -0
	}

	length := len(myout)
	bce := -((1 / float64(length)) * sum)

	return bce
}

func BinaryCrossEntropyDerivative(myout float64, ayout float64, sample int) float64 {
	epsilon := 1e-15
	if myout < epsilon {
		myout = epsilon
	} else if myout > 1-epsilon {
		myout = 1 - epsilon
	}

	result := -((ayout / myout) - ((1 - ayout) / (1 - myout)))
	return result
}
