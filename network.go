// Package gobp implements neural networks with backpropagation in Go
package gobp

import (
	"math/rand"
	"runtime"
	"sync"

	"github.com/goki/mat32"
)

// Unit contains the data for a unit in a neural network
type Unit struct {
	Act float32 // the activation value of the unit
	Net float32 // the net input of the unit
	Err float32 // the error of the unit
}

// Network is a neural network. A network should be created by using a pointer to a struct literal with the desired values and then calling Init.
type Network struct {
	LearningRate         float32        // the rate at which the network learns; this is safe to change at any time
	NumInputs            int            // the number of inputs; this can be changed, but Init must be called after
	NumOutputs           int            // the number of outputs; this can be changed, but Init must be called after
	NumHiddenLayers      int            // the number of hidden layers; this can be changed, but Init must be called after
	NumHiddenUnits       int            // the number of units per hidden layer; this can be changed, but Init must be called after
	NumGoroutines        int            // the number of goroutines used for training the network; this can be changed, but Init must be called after
	ActivationFunc       ActivationFunc // the activation function used for computing the activation values for the hidden layers of the network; this can be changed, but Init must be called after
	OutputActivationFunc ActivationFunc // the activation function used for computing the activation values for the output layer of the network; this can be changed, but Init must be called after

	Inputs  []float32 // the values of the network inputs; these must be set manually everytime the network is trained
	Targets []float32 // the values of the output targets; these must be set manually everytime the network is trained

	Layers  []*Layer  // the layers of the network; these should not be modified by the user
	Units   []Unit    // the layer units, including the input and output units; these should not be set manually -- use Inputs to set the inputs
	Weights []float32 // the values of the weights connecting layers; these should not be set manually

}

// Defaults sets the default parameter values for the neural network for all parameters that have missing values
func (n *Network) Defaults() {
	if n.LearningRate == 0 {
		n.LearningRate = 0.05
	}
	if n.NumGoroutines == 0 {
		n.NumGoroutines = runtime.NumCPU()
	}
	if n.ActivationFunc.Derivative == nil {
		n.ActivationFunc = Rectifier
	}
	if n.OutputActivationFunc.Derivative == nil {
		n.OutputActivationFunc = Logistic
	}
}

// // NewNetwork creates and returns a new neural network with the given number of inputs, number of outputs, number of hidden layers, and number of hidden units per hidden layer
// func NewNetwork(numInputs int, numOutputs int, numHiddenLayers int, numHiddenUnits int) *Network {
// 	n := &Network{
// 		Inputs: make([]float32, numInputs),
// 		// there is one target for every output
// 		Targets: make([]float32, numOutputs),

// 		NumInputs:       numInputs,
// 		NumOutputs:      numOutputs,
// 		NumHiddenLayers: numHiddenLayers,
// 		NumHiddenUnits:  numHiddenUnits,
// 		// there is one unit for every input, one unit for every unit on every hidden layer, and one unit for every output
// 		Units: make([]Unit, numInputs+(numHiddenLayers*numHiddenUnits)+numOutputs),
// 	}
// 	n.Defaults()
// 	// if there are no hidden layers, each input just connects to each output once
// 	if numHiddenLayers == 0 {
// 		n.Weights = make([]float32, numInputs*numOutputs)
// 	} else {
// 		// otherwise, each input (of which there are numInputs) connects to each unit on the first hidden layer (of which there are numUnits), each unit on each hidden layer connects to each unit on the next hidden layer (except for on the final hidden layer, when it connects to each output, and we account for that next, so we subtract one now), and each unit on the last hidden layer (of which there are numUnits) connects to each output (of which there are numOutputs)
// 		n.Weights = make([]float32, (numInputs*numHiddenUnits)+((numHiddenLayers-1)*numHiddenUnits*numHiddenUnits)+(numHiddenUnits*numOutputs))
// 	}
// 	n.InitWeights()
// 	n.InitLayers()
// 	return n
// }

// Init initializes the network based on the values set for the network after calling Defaults.
// It must be called on network creation and every time after any important values of the network are changed.
func (n *Network) Init() {
	n.Defaults()
	n.Inputs = make([]float32, n.NumInputs)
	// there is one target for every output
	n.Targets = make([]float32, n.NumOutputs)
	// there is one unit for every input, one unit for every unit on every hidden layer, and one unit for every output
	n.Units = make([]Unit, n.NumInputs+(n.NumHiddenLayers*n.NumHiddenUnits)+n.NumOutputs)
	// if there are no hidden layers, each input just connects to each output once
	if n.NumHiddenLayers == 0 {
		n.Weights = make([]float32, n.NumInputs*n.NumOutputs)
	} else {
		// otherwise, each input (of which there are numInputs) connects to each unit on the first hidden layer (of which there are numUnits), each unit on each hidden layer connects to each unit on the next hidden layer (except for on the final hidden layer, when it connects to each output, and we account for that next, so we subtract one now), and each unit on the last hidden layer (of which there are numUnits) connects to each output (of which there are numOutputs)
		n.Weights = make([]float32, (n.NumInputs*n.NumHiddenUnits)+((n.NumHiddenLayers-1)*n.NumHiddenUnits*n.NumHiddenUnits)+(n.NumHiddenUnits*n.NumOutputs))
	}
	n.InitWeights()
	n.InitLayers()
}

// InitWeights initializes the weights with random values between 0 and 1, multiplied by 1 over the square root of the number of inputs
func (n *Network) InitWeights() {
	sqrt := mat32.Sqrt(float32(n.NumInputs))
	for i := range n.Weights {
		// rand / sqrt(numInputs) is a good rule for starting weights
		n.Weights[i] = rand.Float32() / sqrt
	}
}

// InitLayers initializes the layers of the network
func (n *Network) InitLayers() {
	// add 2 to account for the input and output layers
	n.Layers = make([]*Layer, n.NumHiddenLayers+2)
	// the current number of units and weights used
	curNumUnits := 0
	curNumWeights := 0
	// li = layer index
	for li := range n.Layers {
		// the number of units on this layer, the layer below this layer, and the layer above this layer
		numUnits, numUnitsBelow := n.NumHiddenUnits, n.NumHiddenUnits
		// the activation function for this layer
		activationFunc := &n.ActivationFunc
		if li == 0 {
			numUnits = n.NumInputs
			numUnitsBelow = -1
		}
		if li == 1 {
			numUnitsBelow = n.NumInputs
		}
		if li == n.NumHiddenLayers+1 {
			numUnits = n.NumOutputs
			activationFunc = &n.OutputActivationFunc
		}
		// the lower and upper bounds for the units contained within this layer
		unitsLower := curNumUnits
		// because re-slicing is open at the upper bound, we do not need to subtract 1 from numUnits
		unitsUpper := curNumUnits + numUnits
		curNumUnits = unitsUpper
		units := n.Units[unitsLower:unitsUpper]

		numGoroutines := n.NumGoroutines
		// each goroutine does len(units)/numGoroutines units, rounded up
		numUnitsPerGoroutine := int(mat32.Ceil(float32(len(units)) / float32(numGoroutines)))
		// if this results in 0 (so we have more goroutines than units), then there will be 1 unit per goroutine and the same number of goroutines as units
		if numUnitsPerGoroutine == 0 {
			numGoroutines = len(units)
			numUnitsPerGoroutine = 1
		}

		// we need to create these variables outside of the condition so that they can be passed to the layer creation, even though they are irrelevant if we are on the first layer
		var weights []float32
		var weightsLower int
		// weights are stored for the layer they connect to, so there are no weights for the first layer
		if li != 0 {
			// the lower and upper bounds for the weights connecting the previous layer to this layer
			weightsLower = curNumWeights
			// similar to above, we do not need to subtract 1 from numUnits
			// there is a weight for each connection between each unit on the previous layer and each unit on this layer
			weightsUpper := curNumWeights + numUnits*numUnitsBelow
			curNumWeights = weightsUpper
			weights = n.Weights[weightsLower:weightsUpper]
		}
		n.Layers[li] = &Layer{
			Index:          li,
			Units:          units,
			Weights:        weights,
			ActivationFunc: activationFunc,

			NumUnits:             numUnits,
			NumGoroutines:        numGoroutines,
			NumUnitsPerGoroutine: numUnitsPerGoroutine,

			UnitsStart:   unitsLower,
			WeightsStart: weightsLower,
		}
	}
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

// // Forward computes the forward propagation pass using the values of the units from 0 to numInputs-1
// func (n *Network) Forward() {
// 	// need to load inputs into units first
// 	for i := 0; i < n.NumInputs; i++ {
// 		// unit index for the input layer
// 		ui := n.UnitIndex(0, i)
// 		// the input that corresponds with this unit
// 		var input float32
// 		// if someone has provided less inputs than there should be, just use 0 for the input
// 		if i >= len(n.Inputs) {
// 			input = 0
// 		} else {
// 			input = n.Inputs[i]
// 		}
// 		// set both the net input and activation value for the input layer unit to the provided input
// 		n.Units[ui] = Unit{Net: input, Act: input}
// 	}
// 	// need to add two to account for input and output layers
// 	// we start from layer 1 because the layers for the first layer (input layer) should already be set by the user in SetInputs
// 	for layer := 1; layer < n.NumHiddenLayers+2; layer++ {
// 		// the number of units in the current layer
// 		// this is normally just the number of units per hidden layer
// 		numHiddenUnits := n.NumHiddenUnits
// 		// the activation function we use for computing the activation value
// 		// this is normally just the standard supplied activation function
// 		activationFunc := n.ActivationFunc
// 		// however, if we are in the final layer (output layer), the number of units is the number of outputs,
// 		// and the activation function is the output activation function
// 		if layer == n.NumHiddenLayers+1 {
// 			numHiddenUnits = n.NumOutputs
// 			activationFunc = n.OutputActivationFunc
// 		}
// 		// the number of units in the layer below the current layer
// 		// this is normally just the number of units per hidden layer
// 		numUnitsBelow := n.NumHiddenUnits
// 		// however, if we are in the layer after the first layer (input layer), the number of units below is the number of inputs
// 		if layer == 1 {
// 			numUnitsBelow = n.NumInputs
// 		}
// 		// a slice containing all of the net inputs for the current layer. this is used with slice activation functions.
// 		netInputs := make([]float32, numHiddenUnits)
// 		for i := 0; i < numHiddenUnits; i++ {
// 			// unit index for the current layer
// 			ui := n.UnitIndex(layer, i)
// 			// the net input for the current layer
// 			var net float32
// 			// we use h instead of j to emphasize that this a layer below (h is before i in the alphabet)
// 			for h := 0; h < numUnitsBelow; h++ {
// 				// the unit index for the layer below
// 				uib := n.UnitIndex(layer-1, h)
// 				// the unit for the layer below
// 				ub := n.Units[uib] // todo: benchmark with & here or not
// 				// the weight index for the weight between the previous layer at h and the current layer at i
// 				wi := n.WeightIndex(layer-1, h, i)
// 				// the weight between the previous layer at h and the current layer at i
// 				w := n.Weights[wi]
// 				// add to the net input for the current unit the activation value for the unit on the previous layer times the connecting weight
// 				net += ub.Act * w
// 			}
// 			// set the net input for the current unit to the summed value
// 			n.Units[ui].Net = net
// 			// set the activation value for the current unit to the value of the activation function called with the net input,
// 			// if and only if the network is using a standard activation function, not a slice function.
// 			if activationFunc.Func != nil {
// 				n.Units[ui].Act = activationFunc.Func(net)
// 			}
// 			// if we are using a slice function, we need to add each net input to the net inputs slice
// 			if activationFunc.SliceFunc != nil {
// 				netInputs[i] = net
// 			}
// 		}
// 		// if we are using a slice function instead of a standard function, then we determine the activation values here
// 		if activationFunc.SliceFunc != nil {
// 			// the activation values for the units on this layer
// 			acts := activationFunc.SliceFunc(netInputs)
// 			for i, act := range acts {
// 				// the unit index for the current layer
// 				ui := n.UnitIndex(layer, i)
// 				// set the activation value for this unit to the computed activation value
// 				n.Units[ui].Act = act
// 			}
// 		}
// 	}
// }

// LoadInputs loads the inputs of the network into the first layer.
// It is called in Forward, so it is unnecessary to call it separately.
func (n *Network) LoadInputs() {
	layer := n.Layers[0]
	for i, input := range n.Inputs {
		layer.Units[i] = Unit{Act: input, Net: input}
	}
}

// ForwardLayer computes the forward propagation pass from the given (lower) layer to the given (higher) layer
func (n *Network) ForwardLayer(from, to *Layer) {
	// the weight group for this layer
	wg := sync.WaitGroup{}
	// we need to wait for each of the goroutines we will create to finish
	wg.Add(to.NumGoroutines)
	// the net inputs for the to layer
	var netInputs []float32
	// only make net inputs if we are using the slice func to save on memory
	if to.ActivationFunc.SliceFunc != nil {
		netInputs = make([]float32, len(to.Units))
	}
	// do does the calculations for the units part of the to layer from start to end
	do := func(start, end int) {
		for i := range to.Units[start:end] {
			// we have to add start to the index so that we are in the right place
			i += start
			// net input for the to layer
			var net float32
			// we use h instead of j to emphasize that this a layer below (h is before i in the alphabet)
			// ub is the unit for the layer below
			for h, ub := range from.Units {
				// the net input is the sum over the previous layer of the activation value for the previous layer times the connecting weight
				net += ub.Act * to.Weights[to.WeightIndex(h, i)]
			}
			to.Units[i].Net = net
			// if we are using a standard activation function, set the activation value to the value resulting from the function
			if to.ActivationFunc.Func != nil {
				to.Units[i].Act = to.ActivationFunc.Func(net)
			} else {
				// otherwise, add the net input to the net inputs slice
				netInputs[i] = net
			}
		}
		wg.Done()
	}
	// the number of units we have done
	numDone := 0
	for i := 0; i < to.NumGoroutines; i++ {
		end := numDone + to.NumUnitsPerGoroutine
		// if we have gone too far, reset back to the end
		if end > len(to.Units) {
			end = len(to.Units)
		}
		go do(numDone, end)
		numDone = end
	}
	wg.Wait()
	// if we are using a slice function instead of a standard function, then we determine the activation values here
	if to.ActivationFunc.SliceFunc != nil {
		// the activation values for the units on this layer
		acts := to.ActivationFunc.SliceFunc(netInputs)
		for i := range to.Units {
			to.Units[i].Act = acts[i]
		}
	}
}

// Forward computes the forward propagation pass using the values of the inputs
func (n *Network) Forward() {
	n.LoadInputs()
	// we start at the bottom and work our way up in Forward; we skip the output layer by subtracting 1 because we add 1 to "to", and we only want the output layer to be the "to" layer
	for i := 0; i < len(n.Layers)-1; i++ {
		// we always go from the current layer to the layer above it in Forward
		n.ForwardLayer(n.Layers[i], n.Layers[i+1])
	}
}

// OutputErrors computes the errors for the units on the output layer by comparing their activation values with the target values
// It returns the total Sum Squared Error (SSE) of all of the output errors
func (n *Network) OutputErrors() float32 {
	var sse float32
	layer := n.Layers[len(n.Layers)-1]
	for i, unit := range layer.Units {
		// error is target minus activation
		err := n.Targets[i] - unit.Act
		layer.Units[i].Err = err
		sse += err * err
	}
	return sse
}

// BackLayer computes the backward error propagation pass from the given (higher) layer to the given (lower) layer
func (n *Network) BackLayer(from, to *Layer) {
	// the weight group for this layer
	wg := sync.WaitGroup{}
	// we need to wait for each of the goroutines we will create to finish
	wg.Add(to.NumGoroutines)
	// do does the calculations for the units part of the to layer from start to end
	do := func(start, end int) {
		// u is the unit for the current layer
		for i, u := range to.Units[start:end] {
			// we have to add start to the index so that we are in the right place
			i += start
			// error for the unit on the to layer
			var err float32
			// we use j instead of h to emphasize that this is a layer above (j is after i in the alphabet)
			// ua is the unit for the layer above
			for j, ua := range from.Units {
				// weight index for the connecting weight
				wi := from.WeightIndex(i, j)
				// the error is the sum over the layer above of the error of the unit above, times the derivative of the activation function at the activation value of the unit above, times the connecting weight
				err += ua.Err * from.ActivationFunc.Derivative(ua.Act) * from.Weights[wi]
				// the delta is the learning rate, times the error of the unit above, times the derivative of the activation function at the activation value of the unit above, times the activation value of the current unit
				del := n.LearningRate * ua.Err * from.ActivationFunc.Derivative(ua.Act) * u.Act
				// apply the delta to the connecting weight
				from.Weights[wi] += del
			}
			to.Units[i].Err = err
		}
		wg.Done()
	}
	// the number of units we have done
	numDone := 0
	for i := 0; i < to.NumGoroutines; i++ {
		end := numDone + to.NumUnitsPerGoroutine
		// if we have gone too far, reset back to the end
		if end > len(to.Units) {
			end = len(to.Units)
		}
		go do(numDone, end)
		numDone = end
	}
	wg.Wait()
}

// Back computes the backward error propagation pass and returns the total Sum Squared Error (SSE) of all of the output errors
func (n *Network) Back() float32 {
	sse := n.OutputErrors()
	// we start at the output layer and work our way down in Back
	for i := len(n.Layers) - 1; i > 0; i-- {
		// we always go from the current layer to the layer below it in Back
		n.BackLayer(n.Layers[i], n.Layers[i-1])
	}
	return sse
}

// // Back computes the backward error propagation pass and returns the cumulative sum squared error (SSE) of all of the errors
// func (n *Network) Back() float32 {
// 	var sse float32
// 	// need to add one to account for input and output layers (ex: numLayers = 0 => we start at layer index 1 (effective length of 2 with >= operator))
// 	for layer := n.NumHiddenLayers + 1; layer >= 0; layer-- {
// 		// numUnits := n.numUnitsPerLayer[layer]
// 		// if we are in the output layer, compute the error directly by comparing each unit with its target
// 		if layer == n.NumHiddenLayers+1 {
// 			for i := 0; i < n.NumOutputs; i++ {
// 				// unit index for the output layer
// 				ui := n.UnitIndex(layer, i)
// 				// error is the target minus the current activation value
// 				err := n.Targets[i] - n.Units[ui].Act
// 				// set the error to what we computed
// 				n.Units[ui].Err = err
// 				// add the error squared to the total sum squared error (SSE)
// 				sse += err * err
// 			}
// 		} else {
// 			// otherwise, we compute the error in relation to higher-up errors

// 			// the number of units in the current layer
// 			// this is normally just the number of units per hidden layer
// 			numHiddenUnits := n.NumHiddenUnits
// 			// however, if we are in the first layer (input layer), the number of units is the number of inputs
// 			if layer == 0 {
// 				numHiddenUnits = n.NumInputs
// 			}
// 			// the number of units in the layer above the current layer
// 			// this is normally just the number of units per layer
// 			numUnitsAbove := n.NumHiddenUnits
// 			// the activation function we use for computing the derivative
// 			// this is normally just the standard supplied activation function
// 			activationFunc := n.ActivationFunc
// 			// however, if we are in the layer before the final layer (output layer), the number of units above is the number of outputs,
// 			// and the activation function is the output activation function
// 			if layer == n.NumHiddenLayers {
// 				numUnitsAbove = n.NumOutputs
// 				activationFunc = n.OutputActivationFunc
// 			}
// 			// i = index for current layer, j = index for layer above
// 			for i := 0; i < numHiddenUnits; i++ {
// 				// unit index for the current layer
// 				ui := n.UnitIndex(layer, i)
// 				// unit for the current layer
// 				u := n.Units[ui]
// 				// total error for this unit (error = sum over j of: error at j * activation func derivative of activation at j * weight between i and j)
// 				var err float32
// 				for j := 0; j < numUnitsAbove; j++ {
// 					// unit index for the layer above
// 					uia := n.UnitIndex(layer+1, j)
// 					// unit for the layer above
// 					ua := n.Units[uia] // todo: benchmark using &
// 					// weight index for current layer to layer above
// 					wi := n.WeightIndex(layer, i, j)
// 					// weight for current layer to layer above
// 					w := n.Weights[wi]
// 					// add to the error for the current unit using the formula specified at the definition of err
// 					err += ua.Err * activationFunc.Derivative(ua.Act) * w
// 					// the delta for this weight (learning rate * error for the unit on the layer above * activation function derivative of activation value for the unit on the above layer * the activation value for the unit on the current layer)
// 					// todo: get rid of lrate here
// 					del := n.LearningRate * ua.Err * activationFunc.Derivative(ua.Act) * u.Act
// 					// apply delta to the weight
// 					n.Weights[wi] += del

// 					// todo: ADAM
// 					// n.Momentum is a parameter = 0.9 default
// 					// n.moment[wi] = n.Momentum * n.moment[wi] + (1 - n.Momentum) * del
// 					// n.var[wi] = n.VarRate * n.var[wi] + (1 - n.VarRate) * del * del
// 					// n.dwt[wi] = n.LearningRate * ??
// 					// todo: use AdaMax instead!!!! much better, doesn't depend on t
// 					// n.weights[wi] += n.dwt[wi]
// 				}
// 				// set the error to the computed error
// 				n.Units[ui].Err = err
// 			}
// 		}
// 	}
// 	return sse
// }

// Outputs returns the output activations of the network
func (n *Network) Outputs() []float32 {
	res := make([]float32, n.NumOutputs)
	layer := n.Layers[len(n.Layers)-1]
	for i, unit := range layer.Units {
		res[i] = unit.Act
	}
	return res
}
