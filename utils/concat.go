package utils

func Concat(feature1 *[][]float64, feature2 *[][]float64) {
	len1 := len(*feature1)
	len2 := len(*feature2)

	if len1 >= len2 {
		for i := range len(*feature1) {
			if i <= len2 {
				(*feature1)[i] = append((*feature1)[i], (*feature2)[i]...)
			} else {
				feature := make([]float64, len((*feature2)[0]))
				(*feature1)[i] = append((*feature1)[i], feature...)
			}
		}
	} else {
		for i := range len(*feature2) {
			if i <= len1 {
				(*feature1)[i] = append((*feature1)[i], (*feature2)[i]...)
			} else {
				feature := make([]float64, len((*feature2)[0]))
				new := append(feature, (*feature2)[i]...)
				*feature1 = append(*feature1, new)
			}
		}
	}
}
