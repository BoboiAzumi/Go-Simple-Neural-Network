package sequential

import (
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/layer"
)

func ChooseLayer(activation string) layer.Layer {
	switch activation {
	case "softmax":
		return layer.NewSoftmaxLayer()
	default:
		return layer.NewLayer()
	}
}
