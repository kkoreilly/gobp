// Package gobp implements neural networks with backpropagation in Go
package gobp

import "log"

// Network is a neural network
type Network struct {
	LearningRate float32   // the rate at which the network learns
	NumLayers    int       // the number of hidden layers
	NumInputs    int       // the number of inputs
	NumOutputs   int       // the number of outputs
	NumUnits     int       // the number of units per hidden layer
	Units        []Unit    // the hidden layer units, including the input and output units
	Weights      []float32 // the values of the weights connecting layers
	Targets      []float32 // the values of the output targets
}

// Unit contains the data for a unit in the neural network
type Unit struct {
	Act float32 // the activation value of the unit
	Net float32 // the net input of the unit
	Err float32 // the error of the unit
}

// Rectifier is the rectifier (ReLU) activation function that returns x if x > 0 and 0 otherwise
func Rectifier(x float32) float32 {
	if x > 0 {
		return x
	}
	return 0
}

// RectifierDerivative is the derivative of the rectifier (ReLU) activation function that returns 1 if x > 0 and 0 otherwise.
func RectifierDerivative(x float32) float32 {
	if x > 0 {
		return 1
	}
	return 0
}

// NewNetwork creates and returns a new network with the given information
func NewNetwork(learningRate float32, numLayers int, numInputs int, numOutputs int, numUnits int) *Network {
	n := &Network{
		LearningRate: learningRate,
		NumLayers:    numLayers,
		NumInputs:    numInputs,
		NumOutputs:   numOutputs,
		NumUnits:     numUnits,
		Units:        make([]Unit, numInputs+(numLayers*numUnits)+numOutputs),
		// each input connects to each unit on the first hidden layer, each unit on each hidden layer connects to each unit on the next hidden layer (except for on the final hidden layer, when it connects to each output, and we account for that next, so we subtract one now), and each unit on the final hidden layer connects to each output
		Weights: make([]float32, (numInputs*numUnits)+((numLayers-1)*numUnits*numUnits)+(numUnits*numOutputs)),
		Targets: make([]float32, numOutputs),
	}
	// need to initialize weights first
	for i := range n.Weights {
		n.Weights[i] = 0.1
	}
	return n
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
	return n.NumInputs + (layer-1)*n.NumUnits + idx
}

// WeightIndex returns the index of the weight originating from the given layer index (layer) at the given index (from) going to the given index on the layer layer+1 (to)
func (n *Network) WeightIndex(layer, from, to int) int {
	// if starting at first layer (input layer), we start out by locating macro-scale by from (multiplied by NumUnits -- like base 10 and multiplying by 10 to change effect of digit, except with base NumUnits), and then we increment within that by to.
	// we multiply by NumUnits instead of NumInputs because we are trying to create subdivisions in which we place index in hidden layer, so those subdivisions need to be as long as there are many units per hidden layer.
	if layer == 0 {
		return n.NumUnits*from + to
	}
	// otherwise, we always start by offsetting by NumUnits * NumInputs, which is the total number of indices we should have filled with the inputs because each input (of which there are NumInputs) has as many hidden units as there are units per hidden layer (NumUnits).
	// then, we add to the offset the connections between the units in the hidden layers; for each hidden layer we have already done (we subtract one to account for the input layer, which we already accounted for), each unit (of which there are NumUnits) should have connected to each unit on the next hidden layer (of which there are also NumUnits), so (layer - 1) * NumUnits * NumUnits.
	// finally, we increment to our position inside of our subdivision (to)
	return (n.NumUnits * n.NumInputs) + ((layer - 1) * n.NumUnits * n.NumUnits) + to
}

// Forward computes the forward propagation pass using the values of the units from 0 to NumInputs-1
func (n *Network) Forward() {
	// first we need to compute the activation values for the user-set input values
	for i := 0; i < n.NumInputs; i++ {
		// unit index for the current unit in the input layer
		ui := n.UnitIndex(0, i)
		// just set the activation to the value of the activation function (rectifier) called with the user-set net input
		n.Units[ui].Act = Rectifier(n.Units[ui].Net)
	}
	// need to add two to account for input and output layers
	// we start from layer 1 because we already computed the values for the first layer (input layer) above
	for layer := 1; layer < n.NumLayers+2; layer++ {
		for i := 0; i < n.NumUnits; i++ {
			// unit index for the current layer
			ui := n.UnitIndex(layer, i)
			// the net input for the current layer
			var net float32
			// the number of units in the layer below the current layer
			// this is normally just the number of units per layer
			numUnitsBelow := n.NumUnits
			// however, if we are in the layer after the first layer (input layer), the number of units below is the number of inputs
			if layer == 1 {
				numUnitsBelow = n.NumInputs
			}
			// we use h instead of j to emphasize that this a layer below (h is before i in the alphabet)
			for h := 0; h < numUnitsBelow; h++ {
				// the unit index for the layer below
				uib := n.UnitIndex(layer-1, h)
				// the unit for the layer below
				ub := n.Units[uib]
				// the weight index for the weight between the previous layer at h and the current layer at i
				wi := n.WeightIndex(layer-1, h, i)
				// the weight between the previous layer at h and the current layer at i
				w := n.Weights[wi]
				// add to the net input for the current unit the activation value for the unit on the previous layer times the connecting weight
				net += ub.Act * w
			}
			// set the net input for the current unit to the summed value
			n.Units[ui].Net = net
			// set the activation value for the current unit to the value of the activation function (rectifier) called with the net input
			n.Units[ui].Act = Rectifier(net)
		}
	}
}

// Back computes the backward error propagation pass and returns the cumulative sum squared error (SSE) of all of the errors
func (n *Network) Back() float32 {
	var sse float32
	// need to add one to account for input and output layers (ex: NumLayers = 0 => we start at layer index 1 (effective length of 2 with >= operator))
	for layer := n.NumLayers + 1; layer >= 0; layer-- {
		// if we are in the output layer, compute the error directly by comparing each unit with its target
		if layer == n.NumLayers+1 {
			for i := 0; i < n.NumOutputs; i++ {
				ui := n.UnitIndex(layer, i)
				err := n.Targets[i] - n.Units[ui].Act
				log.Println(n.Targets[i], n.Units[ui].Act)
				n.Units[ui].Err = err * err
			}
		} else { // otherwise, compute it in relation to higher-up errors
			// the number of units in the layer above the current layer
			// this is normally just the number of units per layer
			numUnitsAbove := n.NumUnits
			// however, if we are in the layer before the final layer (output layer), the number of units above is the number of outputs
			if layer == n.NumLayers {
				numUnitsAbove = n.NumOutputs
			}

			// i = index for current layer, j = index for layer above
			for i := 0; i < n.NumUnits; i++ {
				// unit index for the current layer
				ui := n.UnitIndex(layer, i)
				// unit for the current layer
				u := n.Units[ui]
				// total error for this unit (error = sum over j of: error at j * activation func derivative of net input at j * weight between i and j)
				var err float32
				for j := 0; j < numUnitsAbove; j++ {
					// unit index for the layer above
					uia := n.UnitIndex(layer+1, j)
					// unit for the layer above
					ua := n.Units[uia]
					// weight index for current layer to layer above
					wi := n.WeightIndex(layer, i, j)
					// weight for current layer to layer above
					w := n.Weights[wi]
					// add to the error for the current unit using the formula specified at the definition of err
					err += ua.Err * RectifierDerivative(ua.Net) * w
					// the delta for this weight (learning rate * error for the unit on the layer above * activation function derivative of net input for the unit on the current layer * the activation value for the unit on the current layer)
					del := n.LearningRate * ua.Err * RectifierDerivative(u.Net) * u.Act
					// apply delta to the weight
					n.Weights[wi] += del
				}
				// set the error to the computed error
				n.Units[ui].Err = err
				// add the error to the total error
				sse += err
			}
		}
	}
	return sse
}

// Outputs returns the output activations of the network
func (n *Network) Outputs() []float32 {
	res := make([]float32, n.NumOutputs)
	for i := 0; i < n.NumOutputs; i++ {
		res[i] = n.Units[len(n.Units)-i-1].Act
	}
	return res
}
