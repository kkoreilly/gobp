// Package gobp implements neural networks with backpropagation in Go
package gobp

import (
	"fmt"
)

// Network is a neural network
type Network struct {
	LearningRate         float32        // the rate at which the network learns
	NumLayers            int            // the number of hidden layers
	NumInputs            int            // the number of inputs
	NumOutputs           int            // the number of outputs
	NumUnits             int            // the number of units per hidden layer
	Units                []Unit         // the hidden layer units, including the input and output units
	Weights              []float32      // the values of the weights connecting layers
	Targets              []float32      // the values of the output targets
	ActivationFunc       ActivationFunc // the activation function used for computing the activation values for the hidden layers of the network
	OutputActivationFunc ActivationFunc // the activation function used for computing the activation values for the output layer of the network
}

// Unit contains the data for a unit in the neural network
type Unit struct {
	Act float32 // the activation value of the unit
	Net float32 // the net input of the unit
	Err float32 // the error of the unit
}

// NewNetwork creates and returns a new network with the given information
func NewNetwork(learningRate float32, numLayers int, numInputs int, numOutputs int, numUnits int) *Network {
	n := &Network{
		LearningRate: learningRate,
		NumLayers:    numLayers,
		NumInputs:    numInputs,
		NumOutputs:   numOutputs,
		NumUnits:     numUnits,
		// there is one unit for every input, every unit on every hidden layer, and every output
		Units: make([]Unit, numInputs+(numLayers*numUnits)+numOutputs),
		// there is one target for every output
		Targets:              make([]float32, numOutputs),
		ActivationFunc:       Rectifier,
		OutputActivationFunc: Logistic,
	}
	// if there are no hidden layers, each input just connects to each output once
	if numLayers == 0 {
		n.Weights = make([]float32, numInputs*numOutputs)
	} else {
		// otherwise, each input (of which there are numInputs) connects to each unit on the first hidden layer (of which there are numUnits), each unit on each hidden layer connects to each unit on the next hidden layer (except for on the final hidden layer, when it connects to each output, and we account for that next, so we subtract one now), and each unit on the last hidden layer (of which there are numUnits) connects to each output (of which there are numOutputs)
		n.Weights = make([]float32, (numInputs*numUnits)+((numLayers-1)*numUnits*numUnits)+(numUnits*numOutputs))
	}
	// need to initialize weights
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
	// first we need to initialize some numbers that are dependent on our current layer and the number of hidden layers

	// the number of units in the layer above our current layer
	// this will normally be the number of units per hidden layer
	numUnitsAbove := n.NumUnits
	// unless we are in the layer before the final layer (output layer), in which case there will be NumOutputs units in the layer above
	if layer == n.NumLayers {
		numUnitsAbove = n.NumOutputs
	}
	// the number of units in layer 1 (the second layer)
	// this will normally be the number of units per hidden layer
	numUnitsInLayer1 := n.NumUnits
	// but if there are no hidden layers, it will just be the number of units in the output layer
	if n.NumLayers == 0 {
		numUnitsInLayer1 = n.NumOutputs
	}

	// if starting at first layer (input layer), we start out by locating macro-scale by from (multiplied by numUnitsAbove -- like base 10 and multiplying by 10 to change effect of digit, except with base numUnitsAbove), and then we increment within that by to.
	// we multiply by numUnitsAbove instead of NumInputs or NumUnits because we are trying to create subdivisions in which we place index in the layer above, so those subdivisions need to be as long as there are many units in the layer above.
	if layer == 0 {
		return numUnitsAbove*from + to
	}
	// otherwise, we always start by offsetting by NumInputs * numUnitsInLayer1, which is the total number of indices we should have filled with the inputs because each input (of which there are NumInputs) connects to as many units as there are units in layer 1 (the second layer, the one above the first layer (input layer)).
	// then, we add to the offset the connections between the units in the hidden layers; for each hidden layer we have already done (we subtract one to account for the input layer, which we already accounted for), each unit (of which there are NumUnits) should have connected to each unit on the next hidden layer (of which there are also NumUnits), so (layer - 1) * NumUnits * NumUnits.
	// thirdly, we add to the offset our current offset in the from layer by multiplying by the number of units above. this is similar to what we did above if layer == 0 -- numUnitsAbove is effectively the base with which we are creating subdivisions that we can put to into.
	// finally, we increment to our position inside of our subdivision (to)
	return (n.NumInputs * numUnitsInLayer1) + ((layer - 1) * n.NumUnits * n.NumUnits) + numUnitsAbove*from + to
}

// SetInputs sets the inputs of the network to the given slice of inputs.
// It returns an error if the length of the given inputs does not match the NumInputs field of the network.
func (n *Network) SetInputs(inputs []float32) error {
	if len(inputs) != n.NumInputs {
		return fmt.Errorf("gobp: Network: SetInputs: expected %d inputs, got %d", n.NumInputs, len(inputs))
	}
	for i, input := range inputs {
		// unit index for this input
		ui := n.UnitIndex(0, i)
		n.Units[ui].Net = input
		n.Units[ui].Act = input
	}
	return nil
}

// Forward computes the forward propagation pass using the values of the units from 0 to NumInputs-1
func (n *Network) Forward() {
	// need to add two to account for input and output layers
	// we start from layer 1 because the layers for the first layer (input layer) should already be set by the user in SetInputs
	for layer := 1; layer < n.NumLayers+2; layer++ {
		// the number of units in the current layer
		// this is normally just the number of units per hidden layer
		numUnits := n.NumUnits
		// the activation function we use for computing the activation value
		// this is normally just the standard supplied activation function
		activationFunc := n.ActivationFunc
		// however, if we are in the final layer (output layer), the number of units is the number of outputs,
		// and the activation function is the output activation function
		if layer == n.NumLayers+1 {
			numUnits = n.NumOutputs
			activationFunc = n.OutputActivationFunc
		}
		// the number of units in the layer below the current layer
		// this is normally just the number of units per hidden layer
		numUnitsBelow := n.NumUnits
		// however, if we are in the layer after the first layer (input layer), the number of units below is the number of inputs
		if layer == 1 {
			numUnitsBelow = n.NumInputs
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
			// set the activation value for the current unit to the value of the activation function called with the net input
			n.Units[ui].Act = activationFunc.Func(net)
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
				// unit index for the output layer
				ui := n.UnitIndex(layer, i)
				// error is the current activation value minus the target
				err := n.Units[ui].Act - n.Targets[i]
				// set the error to what we computed
				n.Units[ui].Err = err
				// add the error squared to the total sum squared error (SSE)
				sse += err * err
			}
		} else {
			// otherwise, we compute the error in relation to higher-up errors

			// the number of units in the current layer
			// this is normally just the number of units per hidden layer
			numUnits := n.NumUnits
			// however, if we are in the first layer (input layer), the number of units is the number of inputs
			if layer == 0 {
				numUnits = n.NumInputs
			}
			// the number of units in the layer above the current layer
			// this is normally just the number of units per layer
			numUnitsAbove := n.NumUnits
			// the activation function we use for computing the derivative
			// this is normally just the standard supplied activation function
			activationFunc := n.ActivationFunc
			// however, if we are in the layer before the final layer (output layer), the number of units above is the number of outputs,
			// and the activation function is the output activation function
			if layer == n.NumLayers {
				numUnitsAbove = n.NumOutputs
				activationFunc = n.OutputActivationFunc
			}
			// i = index for current layer, j = index for layer above
			for i := 0; i < numUnits; i++ {
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
					err += ua.Err * activationFunc.Derivative(ua.Net) * w
					// the delta for this weight (learning rate * error for the unit on the layer above * activation function derivative of net input for the unit on the above layer * the activation value for the unit on the current layer)
					del := -n.LearningRate * ua.Err * activationFunc.Derivative(ua.Net) * u.Act
					// apply delta to the weight
					n.Weights[wi] += del
				}
				// set the error to the computed error
				n.Units[ui].Err = err
				// add the error squared to the total sum squared error (SSE)
				sse += err * err
			}
		}
	}
	return sse
}

// Outputs returns the output activations of the network
func (n *Network) Outputs() []float32 {
	res := make([]float32, n.NumOutputs)
	// outputs are stored in the last NumOutputs units, so in effect we just get the activation values for n.Units[len(n.Units)-n.NumOutputs:]
	for i := 0; i < n.NumOutputs; i++ {
		res[i] = n.Units[len(n.Units)-n.NumOutputs+i].Act
	}
	return res
}
