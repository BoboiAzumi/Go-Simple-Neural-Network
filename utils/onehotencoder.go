package utils

import "strings"

type OneHotEncoder struct {
	unique []string
}

func (ohe *OneHotEncoder) UniqueInit() {
	if ohe.unique == nil {
		ohe.unique = []string{}
	}
}

func (ohe *OneHotEncoder) SetUnique(unique []string) {
	ohe.unique = unique
}

func (ohe *OneHotEncoder) GetUnique() []string {
	return ohe.unique
}

func (ohe *OneHotEncoder) FindUnique(unique string) (bool, int) {
	ohe.UniqueInit()
	for i := range len(ohe.unique) {
		if strings.ToUpper(unique) == ohe.unique[i] {
			return true, i
		}
	}
	return false, -1
}

func (ohe *OneHotEncoder) ScanUnique(data [][]string, column int) {
	for i := range data {
		isRegistered, _ := ohe.FindUnique(data[i][column])

		if !isRegistered {
			ohe.unique = append(ohe.unique, strings.ToUpper(data[i][column]))
		}
	}
}

func (ohe *OneHotEncoder) Encoding(data [][]string, column int) [][]float64 {
	result := make([][]float64, len(data))

	for i := range data {
		result[i] = make([]float64, len(ohe.unique))
		isRegistered, index := ohe.FindUnique(data[i][column])

		if isRegistered {
			result[i][index] = 1
		}
	}

	return result
}

func NewOneHotEncoder() *OneHotEncoder {
	return &OneHotEncoder{}
}
