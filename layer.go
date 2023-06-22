package gobp

// Layer represents one layer (input, hidden, or output) of a neural network
type Layer struct {
	Index          int            // the index of this layer in the neural network
	Units          []Unit         // the units on this layer
	Weights        []float32      // the weights connecting from the layer below to this layer
	ActivationFunc ActivationFunc // the activation function for this layer

	NumUnits             int // the number of units on this layer
	NumGoroutines        int // the number of goroutines that this layer runs when training
	NumUnitsPerGoroutine int // the number of units that each goroutine handles for this layer

	UnitsStart   int // the starting index of the units on this layer in the broader network units slice
	WeightsStart int // the starting index of the weights on this layer in the broader network weights slice
}

// WeightIndex returns the weight from the given index on the previous layer to the given index on this layer
func (l *Layer) WeightIndex(from, to int) int {
	// we offset by from multiplied by numUnits, and then we get to final position with to
	return from*l.NumUnits + to
}

// // Forward computes the forward propagation pass for the given layer that is part of the given network
// func (l *Layer) Forward(n *Network) {
// 	// if we are the input layer, just feed the inputs into the units
// 	if l.Index == 0 {
// 		for ui := range l.Units {
// 			input := n.Inputs[ui]
// 			l.Units[ui] = Unit{Net: input, Act: input}
// 		}
// 		return
// 	}
// 	for ui := range l.Units {
// 		// net input for the current layer
// 		net := float32(0)
// 		// we use h instead of j to emphasize that this a layer below (h is before i in the alphabet)
// 		for h := 0; h < l.numUnitsBelow; h++ {
// 			// the unit index for the layer below
// 			uib := n.UnitIndex(l.Index-1, h)
// 		}
// 	}
// }
