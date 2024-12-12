package utils

func Flatten(x [][]float64) []float64 {
	var flat []float64 = []float64{}

	for i := range len(x) {
		flat = append(flat, x[i]...)
	}

	return flat
}
