package optimizer

import (
	"math"
)

type ADAM struct {
	lr      float64
	beta1   float64
	beta2   float64
	epsilon float64
	t       int
}

func (optim *ADAM) UpdateWeight(gradient float64, m *[][]float64, v *[][]float64, i int, j int) float64 {
	(*m)[i][j] = optim.beta1*(*m)[i][j] + (1-optim.beta1)*gradient
	(*v)[i][j] = optim.beta2*(*v)[i][j] + (1-optim.beta2)*gradient*gradient

	mCorrected := (*m)[i][j] / (1 - math.Pow(optim.beta1, float64(optim.t)))
	vCorrected := (*v)[i][j] / (1 - math.Pow(optim.beta2, float64(optim.t)))

	a := (math.Sqrt(vCorrected) + optim.epsilon)

	return optim.lr * mCorrected / a
}

func (optim *ADAM) UpdateBias(gradient float64, m *[]float64, v *[]float64, i int) float64 {
	(*m)[i] = optim.beta1*(*m)[i] + (1-optim.beta1)*gradient
	(*v)[i] = optim.beta2*(*v)[i] + (1-optim.beta2)*gradient*gradient

	mCorrected := (*m)[i] / (1 - math.Pow(optim.beta1, float64(optim.t)))
	vCorrected := (*v)[i] / (1 - math.Pow(optim.beta2, float64(optim.t)))

	return optim.lr * mCorrected / (math.Sqrt(vCorrected) + optim.epsilon)
}

func (optim *ADAM) Info() string {
	return "adam"
}

func (optim *ADAM) Step() {
	optim.t += 1
}

func NewADAM(
	lr float64,
	beta1 float64,
	beta2 float64,
	epsilon float64,
) Optimizer {
	return &ADAM{
		lr:      lr,
		beta1:   beta1,
		beta2:   beta2,
		epsilon: epsilon,
		t:       1,
	}
}
