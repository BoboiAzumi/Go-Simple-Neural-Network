package sequential

import (
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/loss"
)

func ChooseDerivativeLoss(losstype string) loss.DerivativeLossFunc {
	switch losstype {
	case "binarycrossentropy":
		return loss.BinaryCrossEntropyDerivative
	case "categoricalcrossentropy":
		return loss.BinaryCrossEntropyDerivative // Not used actually
	case "mse":
		return loss.MSEDerivative
	default:
		return loss.MSEDerivative
	}
}
