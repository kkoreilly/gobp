// Package gobp implements neural networks with backpropagation in Go
package gobp

// Network is a neural network
type Network struct {
	LearningRate         float32        // the rate at which the network learns; this is safe to change
	ActivationFunc       ActivationFunc // the activation function used for computing the activation values for the hidden layers of the network; the default is Rectifier, but it can be set to anything
	OutputActivationFunc ActivationFunc // the activation function used for computing the activation values for the output layer of the network; the default is Logistic, but it can be set to anything
	Inputs               []float32      // the values of the network inputs; these must be set manually
	Targets              []float32      // the values of the output targets; these must be set manually

	numInputs  int       // the number of inputs; this should not be modified after network creation
	numOutputs int       // the number of outputs; this should not be modified after network creation
	numLayers  int       // the number of hidden layers; this should not be modified after network creation
	numUnits   int       // the number of units per hidden layer; this should not be modified after network creation
	units      []Unit    // the hidden layer units, including the input and output units; these should not be set manually -- use SetInputs to set inputs
	weights    []float32 // the values of the weights connecting layers; these should not be set manually

}

// Unit contains the data for a unit in a neural network
type Unit struct {
	Act float32 // the activation value of the unit
	Net float32 // the net input of the unit
	Err float32 // the error of the unit
}

// NewNetwork creates and returns a new neural network with the given number of inputs, number of outputs, number of hidden layers, and number of units per hidden layer
func NewNetwork(numInputs int, numOutputs int, numLayers int, numUnits int) *Network {
	n := &Network{
		// 0.1 is a decent default learning rate
		LearningRate: 0.1,
		// Rectifier is the default activation function; this should work in almost all circumstances
		ActivationFunc: Rectifier,
		// Logistic is the default output activation function; this is not ideal for things like multi-class classification, but people can easily change it to fit their needs
		OutputActivationFunc: Logistic,
		Inputs:               make([]float32, numInputs),
		// there is one target for every output
		Targets: make([]float32, numOutputs),

		numInputs:  numInputs,
		numOutputs: numOutputs,
		numLayers:  numLayers,
		numUnits:   numUnits,
		// there is one unit for every input, one unit for every unit on every hidden layer, and one unit for every output
		units: make([]Unit, numInputs+(numLayers*numUnits)+numOutputs),
	}
	// if there are no hidden layers, each input just connects to each output once
	if numLayers == 0 {
		n.weights = make([]float32, numInputs*numOutputs)
	} else {
		// otherwise, each input (of which there are numInputs) connects to each unit on the first hidden layer (of which there are numUnits), each unit on each hidden layer connects to each unit on the next hidden layer (except for on the final hidden layer, when it connects to each output, and we account for that next, so we subtract one now), and each unit on the last hidden layer (of which there are numUnits) connects to each output (of which there are numOutputs)
		n.weights = make([]float32, (numInputs*numUnits)+((numLayers-1)*numUnits*numUnits)+(numUnits*numOutputs))
	}
	// need to initialize weights
	for i := range n.weights {
		n.weights[i] = 0.1
	}
	return n
}

// NumInputs returns the number of inputs in the network
func (n *Network) NumInputs() int {
	return n.numInputs
}

// NumOutputs returns the number of outputs in the network
func (n *Network) NumOutputs() int {
	return n.numOutputs
}

// NumLayers returns the number of hidden layers in the network
func (n *Network) NumLayers() int {
	return n.numLayers
}

// NumUnits returns the number of units per hidden layer in the network
func (n *Network) NumUnits() int {
	return n.numUnits
}

// Units returns a copy of all of the units in the network
func (n *Network) Units() []Unit {
	res := make([]Unit, len(n.units))
	copy(res, n.units)
	return res
}

// Weights returns a copy of all of the weights in the network
func (n *Network) Weights() []float32 {
	res := make([]float32, len(n.weights))
	copy(res, n.weights)
	return res
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
	return n.numInputs + (layer-1)*n.numUnits + idx
}

// WeightIndex returns the index of the weight originating from the given layer index (layer) at the given index (from) going to the given index on the layer layer+1 (to)
func (n *Network) WeightIndex(layer, from, to int) int {
	// first we need to initialize some numbers that are dependent on our current layer and the number of hidden layers

	// the number of units in the layer above our current layer
	// this will normally be the number of units per hidden layer
	numUnitsAbove := n.numUnits
	// unless we are in the layer before the final layer (output layer), in which case there will be numOutputs units in the layer above
	if layer == n.numLayers {
		numUnitsAbove = n.numOutputs
	}
	// the number of units in layer 1 (the second layer)
	// this will normally be the number of units per hidden layer
	numUnitsInLayer1 := n.numUnits
	// but if there are no hidden layers, it will just be the number of units in the output layer
	if n.numLayers == 0 {
		numUnitsInLayer1 = n.numOutputs
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
	return (n.numInputs * numUnitsInLayer1) + ((layer - 1) * n.numUnits * n.numUnits) + numUnitsAbove*from + to
}

// Forward computes the forward propagation pass using the values of the units from 0 to numInputs-1
func (n *Network) Forward() {
	// need to load inputs into units first
	for i := 0; i < n.numInputs; i++ {
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
		n.units[ui] = Unit{Net: input, Act: input}
	}
	// need to add two to account for input and output layers
	// we start from layer 1 because the layers for the first layer (input layer) should already be set by the user in SetInputs
	for layer := 1; layer < n.numLayers+2; layer++ {
		// the number of units in the current layer
		// this is normally just the number of units per hidden layer
		numUnits := n.numUnits
		// the activation function we use for computing the activation value
		// this is normally just the standard supplied activation function
		activationFunc := n.ActivationFunc
		// however, if we are in the final layer (output layer), the number of units is the number of outputs,
		// and the activation function is the output activation function
		if layer == n.numLayers+1 {
			numUnits = n.numOutputs
			activationFunc = n.OutputActivationFunc
		}
		// the number of units in the layer below the current layer
		// this is normally just the number of units per hidden layer
		numUnitsBelow := n.numUnits
		// however, if we are in the layer after the first layer (input layer), the number of units below is the number of inputs
		if layer == 1 {
			numUnitsBelow = n.numInputs
		}
		for i := 0; i < numUnits; i++ {
			// unit index for the current layer
			ui := n.UnitIndex(layer, i)
			// the net input for the current layer
			var net float32
			// we use h instead of j to emphasize that this a layer below (h is before i in the alphabet)
			for h := 0; h < numUnitsBelow; h++ {
				// the unit index for the layer below
				uib := n.UnitIndex(layer-1, h)
				// the unit for the layer below
				ub := n.units[uib]
				// the weight index for the weight between the previous layer at h and the current layer at i
				wi := n.WeightIndex(layer-1, h, i)
				// the weight between the previous layer at h and the current layer at i
				w := n.weights[wi]
				// add to the net input for the current unit the activation value for the unit on the previous layer times the connecting weight
				net += ub.Act * w
			}
			// set the net input for the current unit to the summed value
			n.units[ui].Net = net
			// set the activation value for the current unit to the value of the activation function called with the net input
			n.units[ui].Act = activationFunc.Func(net)
		}
	}
}

// Back computes the backward error propagation pass and returns the cumulative sum squared error (SSE) of all of the errors
func (n *Network) Back() float32 {
	var sse float32
	// need to add one to account for input and output layers (ex: numLayers = 0 => we start at layer index 1 (effective length of 2 with >= operator))
	for layer := n.numLayers + 1; layer >= 0; layer-- {
		// if we are in the output layer, compute the error directly by comparing each unit with its target
		if layer == n.numLayers+1 {
			for i := 0; i < n.numOutputs; i++ {
				// unit index for the output layer
				ui := n.UnitIndex(layer, i)
				// error is the current activation value minus the target
				err := n.units[ui].Act - n.Targets[i]
				// set the error to what we computed
				n.units[ui].Err = err
				// add the error squared to the total sum squared error (SSE)
				sse += err * err
			}
		} else {
			// otherwise, we compute the error in relation to higher-up errors

			// the number of units in the current layer
			// this is normally just the number of units per hidden layer
			numUnits := n.numUnits
			// however, if we are in the first layer (input layer), the number of units is the number of inputs
			if layer == 0 {
				numUnits = n.numInputs
			}
			// the number of units in the layer above the current layer
			// this is normally just the number of units per layer
			numUnitsAbove := n.numUnits
			// the activation function we use for computing the derivative
			// this is normally just the standard supplied activation function
			activationFunc := n.ActivationFunc
			// however, if we are in the layer before the final layer (output layer), the number of units above is the number of outputs,
			// and the activation function is the output activation function
			if layer == n.numLayers {
				numUnitsAbove = n.numOutputs
				activationFunc = n.OutputActivationFunc
			}
			// i = index for current layer, j = index for layer above
			for i := 0; i < numUnits; i++ {
				// unit index for the current layer
				ui := n.UnitIndex(layer, i)
				// unit for the current layer
				u := n.units[ui]
				// total error for this unit (error = sum over j of: error at j * activation func derivative of net input at j * weight between i and j)
				var err float32
				for j := 0; j < numUnitsAbove; j++ {
					// unit index for the layer above
					uia := n.UnitIndex(layer+1, j)
					// unit for the layer above
					ua := n.units[uia]
					// weight index for current layer to layer above
					wi := n.WeightIndex(layer, i, j)
					// weight for current layer to layer above
					w := n.weights[wi]
					// add to the error for the current unit using the formula specified at the definition of err
					err += ua.Err * activationFunc.Derivative(ua.Net) * w
					// the delta for this weight (learning rate * error for the unit on the layer above * activation function derivative of net input for the unit on the above layer * the activation value for the unit on the current layer)
					del := -n.LearningRate * ua.Err * activationFunc.Derivative(ua.Net) * u.Act
					// apply delta to the weight
					n.weights[wi] += del

				}
				// set the error to the computed error
				n.units[ui].Err = err
				// add the error squared to the total sum squared error (SSE)
				sse += err * err
			}
		}
	}
	return sse
}

// Outputs returns the output activations of the network
func (n *Network) Outputs() []float32 {
	res := make([]float32, n.numOutputs)
	// outputs are stored in the last numOutputs units, so in effect we just get the activation values for n.units[len(n.units)-n.numOutputs:]
	for i := 0; i < n.numOutputs; i++ {
		res[i] = n.units[len(n.units)-n.numOutputs+i].Act
	}
	return res
}
