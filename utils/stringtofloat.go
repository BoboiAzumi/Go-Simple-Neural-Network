package utils

import "strconv"

func StringToFloat(data [][]string, column int) [][]float64 {
	floatconv := make([][]float64, len(data))

	for i := range len(data) {
		floatconv[i] = make([]float64, 1)

		val, err := strconv.ParseFloat(data[i][column], 64)
		if err != nil {
			floatconv[i][0] = 0.0
			continue
		}

		floatconv[i][0] = val
	}

	return floatconv
}
