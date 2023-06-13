package gobp

import (
	"log"
	"testing"
)

func TestNetwork(t *testing.T) {
	n := NewNetwork(0.1, 1, 2, 2, 2)
	var lastSSE float32
	// tolerance for how much sse can increase in one round
	const tol = 1e-3
	inputs := make([]float32, 2)
	for i := 0; i < 100; i++ {
		if i%2 == 0 {
			inputs[0] = 1
			inputs[1] = 0
			n.Targets[0] = 1
			n.Targets[1] = 0
		} else {
			inputs[0] = 0
			inputs[1] = 1
			n.Targets[0] = 0
			n.Targets[1] = 1
		}
		err := n.SetInputs(inputs)
		if err != nil {
			t.Errorf("error setting network inputs: %v", err)
		}
		n.Forward()
		outputs := n.Outputs()
		log.Println("outputs", outputs)
		if i > 2 {
			if i%2 == 0 {
				if outputs[0] <= outputs[1] {
					t.Errorf("error: input %d is even, but output 0 <= output 1 (%g <= %g)\n", i, outputs[0], outputs[1])
				}
			} else {
				if outputs[1] <= outputs[0] {
					t.Errorf("error: input %d is odd, but output 1 <= output 0 (%g <= %g)\n", i, outputs[1], outputs[0])
				}
			}
		}
		sse := n.Back()
		log.Println("sse", sse)
		if i > 2 && (sse-lastSSE) > tol {
			t.Errorf("error: input %d: sse has increased by more than tol from %g to %g\n", i, lastSSE, sse)
		}
		lastSSE = sse
	}
}

func TestUnitIndex(t *testing.T) {
	n := NewNetwork(0.06, 2, 6, 3, 9)

	// how many times each index has occurred
	indexMap := map[int]int{}

	// add 2 to account for input and output layers
	for layer := 0; layer < n.NumLayers+2; layer++ {
		// the number of units on the current layer
		numUnits := n.NumUnits
		// will be NumInputs/NumOutputs if on input/output layer
		if layer == 0 {
			numUnits = n.NumInputs
		}
		if layer == n.NumLayers+1 {
			numUnits = n.NumOutputs
		}
		for unit := 0; unit < numUnits; unit++ {
			ui := n.UnitIndex(layer, unit)
			indexMap[ui]++
		}
	}

	for i := 0; i < len(n.Units); i++ {
		// each index should only occur once
		if indexMap[i] != 1 {
			t.Errorf("error: unit index %d occurs %d times, when it should occur 1 time", i, indexMap[i])
		}
	}
}

func TestWeightIndex(t *testing.T) {
	n := NewNetwork(0.23, 7, 4, 8, 2)

	// how many times each index has occurred
	indexMap := map[int]int{}

	// only add 1 because there are no weights coming from output layer, so we only need to account for input layer
	for layer := 0; layer < n.NumLayers+1; layer++ {
		// the number of units on the current layer
		numUnits := n.NumUnits
		// will be NumInputs/NumOutputs if on input/output layer
		if layer == 0 {
			numUnits = n.NumInputs
		}
		for i := 0; i < numUnits; i++ {
			// the number of units on the layer above
			numUnitsAbove := n.NumUnits
			// will be NumOutputs if on second to last layer (because it connects to output layer)
			if layer == n.NumLayers {
				numUnitsAbove = n.NumOutputs
			}
			for j := 0; j < numUnitsAbove; j++ {
				wi := n.WeightIndex(layer, i, j)
				indexMap[wi]++
			}
		}
	}

	for i := 0; i < len(n.Weights); i++ {
		// each index should only occur once
		if indexMap[i] != 1 {
			t.Errorf("error: weight index %d occurs %d times, when it should occur 1 time", i, indexMap[i])
		}
	}
}
