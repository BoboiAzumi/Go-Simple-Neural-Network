package sequential

import (
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/activation"
)

func ChooseDerivativeActivation(activationFunc string) activation.DerivativeActivationFunc {
	switch activationFunc {
	case "sigmoid":
		return activation.DerivativeSigmoid
	case "relu":
		return activation.DerivativeRelu
	case "linear":
		return activation.DerivativeLinear
	default:
		return activation.DerivativeRelu
	}
}
