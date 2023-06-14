package gobp

import "github.com/goki/mat32"

// ActivationFunc is a function that turns a given net input into an activation value.
// It contains the actual activation function (Func) and its derivative (Derivative).
type ActivationFunc struct {
	Func       func(x float32) float32
	Derivative func(x float32) float32
}

// Identity is the identity activation function that returns the input unchanged
var Identity = ActivationFunc{
	Func:       IdentityFunc,
	Derivative: IdentityDerivative,
}

// IdentityFunc just returns the input x unchanged
func IdentityFunc(x float32) float32 {
	return x
}

// IdentityDerivative returns the derivative of the identity function (which is 1)
func IdentityDerivative(x float32) float32 {
	return 1
}

// Rectifier is the rectifier (ReLU) activation function that returns x if x > 0 and 0 otherwise
var Rectifier = ActivationFunc{
	Func:       RectifierFunc,
	Derivative: RectifierDerivative,
}

// RectifierFunc returns the value of the rectifier (ReLU) activation function at the given point (x if x > 0 and 0 otherwise)
func RectifierFunc(x float32) float32 {
	if x > 0 {
		return x
	}
	return 0
}

// RectifierDerivative returns the derivative of the rectifier (ReLU) activation function at the given point (1 if x > 0 and 0 otherwise)
func RectifierDerivative(x float32) float32 {
	if x > 0 {
		return 1
	}
	return 0
}

// Logistic is the standard logistic / Sigmoid activation function (1 / (1 + e^-x))
var Logistic = ActivationFunc{
	Func:       LogisticFunc,
	Derivative: LogisticFuncDerivative,
}

// LogisticFunc returns the value of the standard logistic / Sigmoid activation function at the given point (1 / (1 + e^-x))
func LogisticFunc(x float32) float32 {
	return 1 / (1 + mat32.FastExp(-x))
}

// LogisticFuncDerivative returns the derivative of the standard logistic / Sigmoid activation function at the given point
func LogisticFuncDerivative(x float32) float32 {
	return LogisticFunc(x) * (1 - LogisticFunc(x))
}
