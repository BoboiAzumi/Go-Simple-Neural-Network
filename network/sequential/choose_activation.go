package sequential

import (
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/activation"
)

func ChooseActivation(activationFunc string) activation.ActivationFunc {
	switch activationFunc {
	case "sigmoid":
		return activation.Sigmoid
	case "relu":
		return activation.Relu
	case "linear":
		return activation.Linear
	default:
		return activation.DerivativeRelu
	}
}
