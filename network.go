// Package gobp implements neural networks with backpropagation in Go
package gobp

import (
	"fmt"
	"math/rand"

	"github.com/goki/mat32"
)

// Unit contains the data for a unit in a neural network
type Unit struct {
	Act float32 // the activation value of the unit
	Net float32 // the net input of the unit
	Err float32 // the error of the unit
}

// Network is a neural network
type Network struct {
	LearningRate         float32        // the rate at which the network learns; this is safe to change
	WeightVariance       float32        // variance in the initial weight values
	ActivationFunc       ActivationFunc // the activation function used for computing the activation values for the hidden layers of the network; the default is Rectifier, but it can be set to anything
	OutputActivationFunc ActivationFunc // the activation function used for computing the activation values for the output layer of the network; the default is Logistic, but it can be set to anything
	Inputs               []float32      // the values of the network inputs; these must be set manually
	Targets              []float32      // the values of the output targets; these must be set manually
	Layers               []Layer

	// numUnitsPerLayer []int
	NumInputs       int       // the number of inputs; this should not be modified after network creation
	NumOutputs      int       // the number of outputs; this should not be modified after network creation
	NumHiddenLayers int       // the number of hidden layers; this should not be modified after network creation
	NumHiddenUnits  int       // the number of units per hidden layer; this should not be modified after network creation
	Units           []Unit    // the hidden layer units, including the input and output units; these should not be set manually -- use SetInputs to set inputs
	Weights         []float32 // the values of the weights connecting layers; these should not be set manually

}

// Defaults sets the default parameter values for the neural network
func (n *Network) Defaults() {
	// 0.05 is a decent default learning rate
	n.LearningRate = 0.05
	n.WeightVariance = 0.1
	// Rectifier is the default activation function; this should work in almost all circumstances
	n.ActivationFunc = Rectifier
	// Logistic is the default output activation function; this is not ideal for things like multi-class classification, but people can easily change it to fit their needs
	n.OutputActivationFunc = Logistic
}

// NewNetwork creates and returns a new neural network with the given number of inputs, number of outputs, number of hidden layers, and number of hidden units per hidden layer
func NewNetwork(numInputs int, numOutputs int, numHiddenLayers int, numHiddenUnits int) *Network {
	n := &Network{
		Inputs: make([]float32, numInputs),
		// there is one target for every output
		Targets: make([]float32, numOutputs),

		NumInputs:       numInputs,
		NumOutputs:      numOutputs,
		NumHiddenLayers: numHiddenLayers,
		NumHiddenUnits:  numHiddenUnits,
		// there is one unit for every input, one unit for every unit on every hidden layer, and one unit for every output
		Units: make([]Unit, numInputs+(numHiddenLayers*numHiddenUnits)+numOutputs),
	}
	n.Defaults()
	// if there are no hidden layers, each input just connects to each output once
	if numHiddenLayers == 0 {
		n.Weights = make([]float32, numInputs*numOutputs)
	} else {
		// otherwise, each input (of which there are numInputs) connects to each unit on the first hidden layer (of which there are numUnits), each unit on each hidden layer connects to each unit on the next hidden layer (except for on the final hidden layer, when it connects to each output, and we account for that next, so we subtract one now), and each unit on the last hidden layer (of which there are numUnits) connects to each output (of which there are numOutputs)
		n.Weights = make([]float32, (numInputs*numHiddenUnits)+((numHiddenLayers-1)*numHiddenUnits*numHiddenUnits)+(numHiddenUnits*numOutputs))
	}
	n.InitWeights()
	n.InitLayers()
	return n
}

// InitWeights initializes the weights with random values between 0 and 1, multiplied by WeightVariance
func (n *Network) InitWeights() {
	for i := range n.Weights {
		// rand / sqrt(numInputs) is a good rule for starting weights
		n.Weights[i] = rand.Float32() / mat32.Sqrt(float32(n.NumInputs))
	}
}

// InitLayers initializes the layers of the network
func (n *Network) InitLayers() {
	// add 2 to account for the input and output layers
	n.Layers = make([]Layer, n.NumHiddenLayers+2)
	// li = layer index
	for li := range n.Layers {
		// the number of units on this layer, the layer below this layer, and the layer above this layer
		numUnits, numUnitsBelow, numUnitsAbove := n.NumHiddenUnits, n.NumHiddenUnits, n.NumHiddenUnits
		if li == 0 {
			numUnits = n.NumInputs
			numUnitsBelow = -1
		}
		if li == 1 {
			numUnitsBelow = n.NumInputs
		}
		if li == n.NumHiddenLayers {
			numUnitsAbove = n.NumOutputs
		}
		if li == n.NumHiddenLayers+1 {
			numUnits = n.NumOutputs
			numUnitsAbove = -1
		}
		// the lower and upper bounds for the units contained within this layer
		unitsLower := n.UnitIndex(li, 0)
		// because re-slicing is open at the upper bound, we do not need to subtract 1 from numUnits
		unitsUpper := n.UnitIndex(li, numUnits)
		units := n.Units[unitsLower:unitsUpper]
		var weights []float32
		// weights are stored for the layer they connect to, so there are no weights for the first layer
		if li != 0 {
			// similar to above, we do not need to subtract 1 from numUnits, but we do need to subtract 1 from numUnitsBelow because re-slicing only cancels out the final 1 index, not the from offset
			weightsLower := n.WeightIndex(li-1, 0, 0)
			weightsUpper := n.WeightIndex(li-1, numUnitsBelow-1, numUnits)
			weights = n.Weights[weightsLower:weightsUpper]
		}
		n.Layers[li] = Layer{
			Index:   li,
			Units:   units,
			Weights: weights,

			numUnits:      numUnits,
			numUnitsBelow: numUnitsBelow,
			numUnitsAbove: numUnitsAbove,
		}
	}
	fmt.Println(n.Layers)
}

// UnitIndex returns the index of the value on the layer at the given layer index (layer) at the given index on the layer (idx)
func (n *Network) UnitIndex(layer, idx int) int {
	// if in first layer (input layer), just return index because there are no other layers before it
	if layer == 0 {
		return idx
	}
	// otherwise, add the number of inputs to account for the first layer (input layer),
	// then add number of units per hidden layer for each hidden layer we have (need to subtract one to account for first layer being input layer),
	// then finally add index for our current layer (don't care about number of outputs because we will never be above output layer, only on or below it)
	return n.NumInputs + (layer-1)*n.NumHiddenUnits + idx
}

// WeightIndex returns the index of the weight originating from the given layer index (layer) at the given index (from) going to the given index on the layer layer+1 (to)
func (n *Network) WeightIndex(layer, from, to int) int {
	// first we need to initialize some numbers that are dependent on our current layer and the number of hidden layers

	// the number of units in the layer above our current layer
	// this will normally be the number of units per hidden layer
	numUnitsAbove := n.NumHiddenUnits
	// unless we are in the layer before the final layer (output layer), in which case there will be numOutputs units in the layer above
	if layer == n.NumHiddenLayers {
		numUnitsAbove = n.NumOutputs
	}
	// the number of units in layer 1 (the second layer)
	// this will normally be the number of units per hidden layer
	numUnitsInLayer1 := n.NumHiddenUnits
	// but if there are no hidden layers, it will just be the number of units in the output layer
	if n.NumHiddenLayers == 0 {
		numUnitsInLayer1 = n.NumOutputs
	}

	// if starting at first layer (input layer), we start out by locating macro-scale by from (multiplied by numUnitsAbove -- like base 10 and multiplying by 10 to change effect of digit, except with base numUnitsAbove), and then we increment within that by to.
	// we multiply by numUnitsAbove instead of numInputs or numUnits because we are trying to create subdivisions in which we place index in the layer above, so those subdivisions need to be as long as there are many units in the layer above.
	if layer == 0 {
		return numUnitsAbove*from + to
	}
	// otherwise, we always start by offsetting by numInputs * numUnitsInLayer1, which is the total number of indices we should have filled with the inputs because each input (of which there are numInputs) connects to as many units as there are units in layer 1 (the second layer, the one above the first layer (input layer)).
	// then, we add to the offset the connections between the units in the hidden layers; for each hidden layer we have already done (we subtract one to account for the input layer, which we already accounted for), each unit (of which there are numUnits) should have connected to each unit on the next hidden layer (of which there are also numUnits), so (layer - 1) * numUnits * numUnits.
	// thirdly, we add to the offset our current offset in the from layer by multiplying by the number of units above. this is similar to what we did above if layer == 0 -- numUnitsAbove is effectively the base with which we are creating subdivisions that we can put to into.
	// finally, we increment to our position inside of our subdivision (to)
	return (n.NumInputs * numUnitsInLayer1) + ((layer - 1) * n.NumHiddenUnits * n.NumHiddenUnits) + numUnitsAbove*from + to
}

// Forward computes the forward propagation pass using the values of the units from 0 to numInputs-1
func (n *Network) Forward() {
	// need to load inputs into units first
	for i := 0; i < n.NumInputs; i++ {
		// unit index for the input layer
		ui := n.UnitIndex(0, i)
		// the input that corresponds with this unit
		var input float32
		// if someone has provided less inputs than there should be, just use 0 for the input
		if i >= len(n.Inputs) {
			input = 0
		} else {
			input = n.Inputs[i]
		}
		// set both the net input and activation value for the input layer unit to the provided input
		n.Units[ui] = Unit{Net: input, Act: input}
	}
	// need to add two to account for input and output layers
	// we start from layer 1 because the layers for the first layer (input layer) should already be set by the user in SetInputs
	for layer := 1; layer < n.NumHiddenLayers+2; layer++ {
		// the number of units in the current layer
		// this is normally just the number of units per hidden layer
		numHiddenUnits := n.NumHiddenUnits
		// the activation function we use for computing the activation value
		// this is normally just the standard supplied activation function
		activationFunc := n.ActivationFunc
		// however, if we are in the final layer (output layer), the number of units is the number of outputs,
		// and the activation function is the output activation function
		if layer == n.NumHiddenLayers+1 {
			numHiddenUnits = n.NumOutputs
			activationFunc = n.OutputActivationFunc
		}
		// the number of units in the layer below the current layer
		// this is normally just the number of units per hidden layer
		numUnitsBelow := n.NumHiddenUnits
		// however, if we are in the layer after the first layer (input layer), the number of units below is the number of inputs
		if layer == 1 {
			numUnitsBelow = n.NumInputs
		}
		// a slice containing all of the net inputs for the current layer. this is used with slice activation functions.
		netInputs := make([]float32, numHiddenUnits)
		for i := 0; i < numHiddenUnits; i++ {
			// unit index for the current layer
			ui := n.UnitIndex(layer, i)
			// the net input for the current layer
			var net float32
			// we use h instead of j to emphasize that this a layer below (h is before i in the alphabet)
			for h := 0; h < numUnitsBelow; h++ {
				// the unit index for the layer below
				uib := n.UnitIndex(layer-1, h)
				// the unit for the layer below
				ub := n.Units[uib] // todo: benchmark with & here or not
				// the weight index for the weight between the previous layer at h and the current layer at i
				wi := n.WeightIndex(layer-1, h, i)
				// the weight between the previous layer at h and the current layer at i
				w := n.Weights[wi]
				// add to the net input for the current unit the activation value for the unit on the previous layer times the connecting weight
				net += ub.Act * w
			}
			// set the net input for the current unit to the summed value
			n.Units[ui].Net = net
			// set the activation value for the current unit to the value of the activation function called with the net input,
			// if and only if the network is using a standard activation function, not a slice function.
			if activationFunc.Func != nil {
				n.Units[ui].Act = activationFunc.Func(net)
			}
			// if we are using a slice function, we need to add each net input to the net inputs slice
			if activationFunc.SliceFunc != nil {
				netInputs[i] = net
			}
		}
		// if we are using a slice function instead of a standard function, then we determine the activation values here
		if activationFunc.SliceFunc != nil {
			// the activation values for the units on this layer
			acts := activationFunc.SliceFunc(netInputs)
			for i, act := range acts {
				// the unit index for the current layer
				ui := n.UnitIndex(layer, i)
				// set the activation value for this unit to the computed activation value
				n.Units[ui].Act = act
			}
		}
	}
}

// ForwardLayer computes the forward propagation pass from the given layer to the given layer
func (n *Network) ForwardLayer(from Layer, to Layer) {}

// Back computes the backward error propagation pass and returns the cumulative sum squared error (SSE) of all of the errors
func (n *Network) Back() float32 {
	var sse float32
	// need to add one to account for input and output layers (ex: numLayers = 0 => we start at layer index 1 (effective length of 2 with >= operator))
	for layer := n.NumHiddenLayers + 1; layer >= 0; layer-- {
		// numUnits := n.numUnitsPerLayer[layer]
		// if we are in the output layer, compute the error directly by comparing each unit with its target
		if layer == n.NumHiddenLayers+1 {
			for i := 0; i < n.NumOutputs; i++ {
				// unit index for the output layer
				ui := n.UnitIndex(layer, i)
				// error is the target minus the current activation value
				err := n.Targets[i] - n.Units[ui].Act
				// set the error to what we computed
				n.Units[ui].Err = err
				// add the error squared to the total sum squared error (SSE)
				sse += err * err
			}
		} else {
			// otherwise, we compute the error in relation to higher-up errors

			// the number of units in the current layer
			// this is normally just the number of units per hidden layer
			numHiddenUnits := n.NumHiddenUnits
			// however, if we are in the first layer (input layer), the number of units is the number of inputs
			if layer == 0 {
				numHiddenUnits = n.NumInputs
			}
			// the number of units in the layer above the current layer
			// this is normally just the number of units per layer
			numUnitsAbove := n.NumHiddenUnits
			// the activation function we use for computing the derivative
			// this is normally just the standard supplied activation function
			activationFunc := n.ActivationFunc
			// however, if we are in the layer before the final layer (output layer), the number of units above is the number of outputs,
			// and the activation function is the output activation function
			if layer == n.NumHiddenLayers {
				numUnitsAbove = n.NumOutputs
				activationFunc = n.OutputActivationFunc
			}
			// i = index for current layer, j = index for layer above
			for i := 0; i < numHiddenUnits; i++ {
				// unit index for the current layer
				ui := n.UnitIndex(layer, i)
				// unit for the current layer
				u := n.Units[ui]
				// total error for this unit (error = sum over j of: error at j * activation func derivative of activation at j * weight between i and j)
				var err float32
				for j := 0; j < numUnitsAbove; j++ {
					// unit index for the layer above
					uia := n.UnitIndex(layer+1, j)
					// unit for the layer above
					ua := n.Units[uia] // todo: benchmark using &
					// weight index for current layer to layer above
					wi := n.WeightIndex(layer, i, j)
					// weight for current layer to layer above
					w := n.Weights[wi]
					// add to the error for the current unit using the formula specified at the definition of err
					err += ua.Err * activationFunc.Derivative(ua.Act) * w
					// the delta for this weight (learning rate * error for the unit on the layer above * activation function derivative of activation value for the unit on the above layer * the activation value for the unit on the current layer)
					// todo: get rid of lrate here
					del := n.LearningRate * ua.Err * activationFunc.Derivative(ua.Act) * u.Act
					// apply delta to the weight
					n.Weights[wi] += del

					// todo: ADAM
					// n.Momentum is a parameter = 0.9 default
					// n.moment[wi] = n.Momentum * n.moment[wi] + (1 - n.Momentum) * del
					// n.var[wi] = n.VarRate * n.var[wi] + (1 - n.VarRate) * del * del
					// n.dwt[wi] = n.LearningRate * ??
					// todo: use AdaMax instead!!!! much better, doesn't depend on t
					// n.weights[wi] += n.dwt[wi]
				}
				// set the error to the computed error
				n.Units[ui].Err = err
			}
		}
	}
	return sse
}

// Outputs returns the output activations of the network
func (n *Network) Outputs() []float32 {
	res := make([]float32, n.NumOutputs)
	// outputs are stored in the last numOutputs units, so in effect we just get the activation values for n.units[len(n.units)-n.numOutputs:]
	for i := 0; i < n.NumOutputs; i++ {
		res[i] = n.Units[len(n.Units)-n.NumOutputs+i].Act
	}
	return res
}
