package gobp

// Layer represents one layer (input, hidden, or output) of a neural network
type Layer struct {
	Index   int       // the index of this layer in the neural network
	Units   []Unit    // the units on this layer
	Weights []float32 // the weights connecting from the layer below to this layer

	numUnits      int // the number of units on this layer
	numUnitsBelow int // the number of units on the layer below this layer
	numUnitsAbove int // the number of units on the layer above this layer
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
