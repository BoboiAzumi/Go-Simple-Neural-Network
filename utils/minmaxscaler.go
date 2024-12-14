package utils

import (
	"strconv"
)

type MinMaxScaler struct {
	min float64
	max float64
}

func (mm *MinMaxScaler) Fit(data [][]string, column int) {
	for i := range len(data) {
		val, err := strconv.ParseFloat(data[i][column], 64)
		if err != nil {
			panic("Not contains any number")
		}

		if i == 0 {
			mm.min = val
			mm.max = val
		}

		if val >= mm.max && i != 0 {
			mm.max = val
		}

		if val <= mm.min && i != 0 {
			mm.min = val
		}
	}
}

func (mm *MinMaxScaler) Transform(data [][]string, column int) [][]float64 {
	z := make([][]float64, len(data))
	for i := range len(data) {
		val, err := strconv.ParseFloat(data[i][column], 64)
		if err != nil {
			panic("Not contains any number")
		}

		z[i] = make([]float64, 1)
		z[i][0] = (val - mm.min) / (mm.max - mm.min)
	}

	return z
}

func NewMinMaxScaler() *MinMaxScaler {
	return &MinMaxScaler{}
}
