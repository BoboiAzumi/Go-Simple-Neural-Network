package loss

type LossFunc func(myout []float64, ayout []float64) float64
type DerivativeLossFunc func(myout float64, ayout float64) float64
