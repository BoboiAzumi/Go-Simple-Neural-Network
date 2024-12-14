package utils

import (
	"math"
	"strconv"
)

type StandardScaler struct {
	mean float64
	std  float64
}

func (ss *StandardScaler) Fit(data [][]string, column int) {
	sum := 0.0
	for i := range len(data) {
		val, err := strconv.ParseFloat(data[i][column], 64)
		if err != nil {
			panic("Not contains any number")
		}

		sum += val
	}
	ss.mean = sum / float64(len(data))

	sum = 0.0
	for i := range len(data) {
		val, err := strconv.ParseFloat(data[i][column], 64)
		if err != nil {
			panic("Not contains any number")
		}

		sum += math.Pow(val-ss.mean, 2)
	}

	ss.std = sum / float64(len(data))
}

func (ss *StandardScaler) Transform(data [][]string, column int) [][]float64 {
	z := make([][]float64, len(data))
	for i := range len(data) {
		val, err := strconv.ParseFloat(data[i][column], 64)
		if err != nil {
			panic("Not contains any number")
		}

		z[i] = make([]float64, 1)
		z[i][0] = (val - ss.mean) / ss.std
	}

	return z
}

func NewStandardScaler() *StandardScaler {
	return &StandardScaler{}
}
