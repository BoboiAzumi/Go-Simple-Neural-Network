package optimizer

type SGD struct {
	lr float64
}

func (optim *SGD) UpdateWeight(gradient float64, m *[][]float64, v *[][]float64, i int, j int) float64 {
	return optim.lr * gradient
}

func (optim *SGD) UpdateBias(gradient float64, m *[]float64, v *[]float64, i int) float64 {
	return optim.lr * gradient
}

func (optim *SGD) Info() string {
	return "sgd"
}

func (optim *SGD) Step() {
	return
}

func NewSGD(lr float64) Optimizer {
	return &SGD{lr: lr}
}
