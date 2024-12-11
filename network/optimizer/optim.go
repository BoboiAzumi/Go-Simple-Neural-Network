package optimizer

type Optimizer interface {
	UpdateWeight(gradient float64, m *[][]float64, v *[][]float64, i int, j int) float64
	UpdateBias(gradient float64, m *[]float64, v *[]float64, i int) float64
	Info() string
	Step()
}
