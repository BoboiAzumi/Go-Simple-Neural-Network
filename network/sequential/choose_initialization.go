package sequential

import weightinitialization "github.com/BoboiAzumi/Go-Simple-Neural-Network/network/weight_initialization"

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
