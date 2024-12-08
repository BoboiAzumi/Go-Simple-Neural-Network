package datasets

// Source : https://www.kaggle.com/datasets/priy998/golf-play-dataset
func GolfXFeatureSoftmax() [][]float64 {
	// Columns mapping for x:
	// outlook_sunny, outlook_rainy, outlook_overcast,
	// temperature_hot, temperature_mild, temperature_cool,
	// humidity_high, humidity_normal, windy

	return [][]float64{
		{0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0},
		{0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
		{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0},
		{1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0},
		{0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0},
		{0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0},
		{1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0},
		{0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0},
		{1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0},
	}
}

func GolfYFeatureSoftmax() [][]float64 {
	// Columns mapping for y:
	// yes or no
	return [][]float64{
		{0.0, 1.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 0.0},
		{1.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 0.0},
		{0.0, 1.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{0.0, 1.0},
	}
}

func GolfXTestSoftmax() [][]float64 {
	return [][]float64{
		{0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0}, // Synthetic sunny, hot, normal, not windy
		{1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0}, // Synthetic rainy, mild, high, windy
		{0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0}, // Synthetic overcast, cool, high, not windy
	}
}

func GolfYTestSoftmax() [][]float64 {
	return [][]float64{
		{0.0, 1.0}, // Expected: no
		{1.0, 0.0}, // Expected: yes
		{1.0, 0.0}, // Expected: yes
	}
}
