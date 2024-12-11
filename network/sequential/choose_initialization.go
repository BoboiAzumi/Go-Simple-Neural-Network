package sequential

import weightinitialization "nn/network/weight_initialization"

func ChooseInitialization(activation string) weightinitialization.Initialization {
	switch activation {
	case "sigmoid":
		return weightinitialization.XavierInitialization
	case "softmax":
		return weightinitialization.XavierInitialization
	case "relu":
		return weightinitialization.HeInitialization
	case "linear":
		return weightinitialization.HeInitialization
	default:
		return weightinitialization.RandomInitialization
	}
}
