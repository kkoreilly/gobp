package gobp

import "github.com/goki/mat32"

// ActivationFunc is a function that turns a given net input into an activation value.
type ActivationFunc struct {
	// Func is the activation function in terms of the net input x. Only one of Func and SliceFunc should be set.
	// Func should be used in most cases instead of SliceFunc unless the activation function is dependent on other inputs.
	Func func(x float32) float32
	// SliceFunc is an alternative activation function that returns a slice of activation values based on a slice of net input values.
	// Only one of Func and SliceFunc should be set; SliceFunc should be used with things like SoftMax that are based on the other inputs.
	SliceFunc func(xs []float32) []float32
	// Derivative is the derivative of the activation function in terms of the result of the actual activation function y.
	// Expressing the derivative as a function of the activation value improves performance in some cases.
	Derivative func(y float32) float32
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

// IdentityDerivative returns the derivative of the identity function (which is always 1, regardless of the activation value y)
func IdentityDerivative(y float32) float32 {
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

// RectifierDerivative returns the derivative of the rectifier (ReLU) activation function at the given activation value y (1 if y > 0 and 0 otherwise).
// Because x > 0 guarantees y > 0, this gives the same results as the way the derivative would traditionally be expressed (as 1 if x > 0 and 0 otherwise).
func RectifierDerivative(y float32) float32 {
	if y > 0 {
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

// LogisticFuncDerivative returns the derivative of the standard logistic / Sigmoid activation function at the given activation value y (y * (1 - y))
func LogisticFuncDerivative(y float32) float32 {
	return y * (1 - y)
}

// SoftMax is the soft arg max activation function
var SoftMax = ActivationFunc{
	SliceFunc:  SoftMaxFunc,
	Derivative: SoftMaxDerivative,
}

// SoftMaxFunc returns the soft arg max function called on the given inputs.
func SoftMaxFunc(xs []float32) []float32 {
	res := make([]float32, len(xs))
	var sum float32
	max := xs[0]
	for _, x := range xs {
		if x > max {
			max = x
		}
	}
	for i, x := range xs {
		res[i] = mat32.FastExp(x - max)
		sum += res[i]
	}
	for i := range xs {
		res[i] /= sum
	}
	return res
}

// SoftMaxDerivative returns the derivative of the soft max function at the given activation value y (which is just 1)
func SoftMaxDerivative(y float32) float32 {
	return 1
}
