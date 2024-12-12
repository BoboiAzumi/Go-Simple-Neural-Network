package sequential

import "github.com/BoboiAzumi/Go-Simple-Neural-Network/network/optimizer"

func ChooseOptimizer(opt string, param []float64) optimizer.Optimizer {
	switch opt {
	case "adam":
		return optimizer.NewADAM(param[0], param[1], param[2], param[3])
	default:
		return optimizer.NewSGD(param[0])
	}
}
