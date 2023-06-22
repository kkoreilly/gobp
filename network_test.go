package gobp

import (
	"fmt"
	"testing"
)

func TestEvenOdd(t *testing.T) {
	n := &Network{
		LearningRate:    0.5,
		NumInputs:       2,
		NumOutputs:      2,
		NumHiddenLayers: 1,
		NumHiddenUnits:  2,
	}
	n.Init()
	var lastSSE float32
	// tolerance for how much sse can increase in one round
	const tol = 0.1
	for i := 0; i < 1000; i++ {
		if i%2 == 0 {
			n.Inputs[0] = 1
			n.Inputs[1] = 0
			n.Targets[0] = 1
			n.Targets[1] = 0
		} else {
			n.Inputs[0] = 0
			n.Inputs[1] = 1
			n.Targets[0] = 0
			n.Targets[1] = 1
		}
		n.Forward()
		outputs := n.Outputs()
		// only check for incorrect outputs after 10 epochs
		if i > 10 {
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
		// fmt.Println("Epoch", i, "sse", sse)
		// only check for sse increases after 100 epochs
		if i > 100 && (sse-lastSSE) > tol {
			t.Errorf("error: input %d: sse has increased by more than tol from %g to %g\n", i, lastSSE, sse)
		}
		lastSSE = sse
	}
}

func TestXOR(t *testing.T) {
	// this test doesn't work yet because we haven't implemented momentum
	// TODO: run this test
	t.SkipNow()
	n := &Network{
		LearningRate:         0.01,
		NumInputs:            2,
		NumOutputs:           1,
		NumHiddenLayers:      1,
		NumHiddenUnits:       10,
		OutputActivationFunc: Rectifier,
	}
	n.Init()
	// the last sse value for each possible set of inputs
	lastSSE := make([]float32, 4)
	test := func(i int, inputType int) {
		n.Forward()
		sse := n.Back()
		if i > 2 && (sse-lastSSE[inputType]) > defTol {
			t.Errorf("error: epoch %d: sse has increased by more than tol from %g to %g\n", i, lastSSE[inputType], sse)
		}
		lastSSE[inputType] = sse
		outputs := n.Outputs()
		fmt.Printf("%d\t%d\t%g\n", i, inputType, sse)
		if i > 100 && !aboutEqual(outputs[0], n.Targets[0], 0.5) {
			t.Errorf("error: epoch %d: inputs %v should result in %g, not %g", i, n.Inputs, n.Targets, outputs)
		}
	}
	for i := 0; i < 1000; i++ {
		n.Inputs[0] = 0
		n.Inputs[1] = 0
		n.Targets[0] = 0
		test(i, 0)

		n.Inputs[0] = 1
		n.Inputs[1] = 0
		n.Targets[0] = 1
		test(i, 1)

		n.Inputs[0] = 0
		n.Inputs[1] = 1
		n.Targets[0] = 1
		test(i, 2)

		n.Inputs[0] = 1
		n.Inputs[1] = 1
		n.Targets[0] = 0
		test(i, 3)
	}

}

func TestSoftMaxEven3(t *testing.T) {
	n := &Network{
		LearningRate: 0.5,
		// four different possibilities; even and multiple of 3, even only, multiple of 3 only, and none
		NumInputs:            4,
		NumOutputs:           4,
		NumHiddenLayers:      1,
		NumHiddenUnits:       10,
		OutputActivationFunc: SoftMax,
	}
	n.Init()
	// the last sse value for each possible set of inputs
	lastSSE := make([]float32, 4)

	for i := 0; i < 1000; i++ {
		// what type of input we have
		inputType := 0
		switch {
		case i%2 == 0 && i%3 == 0:
			inputType = 0
		case i%2 == 0:
			inputType = 1
		case i%3 == 0:
			inputType = 2
		default:
			inputType = 3
		}
		// set the input and target corresponding with our input type to 1 and everything else to 0
		for i := range n.Inputs {
			n.Inputs[i] = 0
			n.Targets[i] = 0
		}
		n.Inputs[inputType] = 1
		n.Targets[inputType] = 1

		n.Forward()
		sse := n.Back()
		// only check for sse increases after 10 epochs
		if i > 10 && (sse-lastSSE[inputType]) > defTol {
			t.Errorf("error: epoch %d: sse has increased by more than tol from %g to %g\n", i, lastSSE[inputType], sse)
		}
		lastSSE[inputType] = sse
		outputs := n.Outputs()
		// only check for incorrect outputs after 100 epochs
		if i > 100 {
			// get the highest value in the outputs and the index storing it
			var max float32
			var maxIdx int
			for j, output := range outputs {
				if output > max {
					max = output
					maxIdx = j
				}
			}
			// the highest result index should be the same as the input type because we set the input with the index of the input type to 1
			if maxIdx != inputType {
				t.Errorf("error: epoch %d: inputs %v should result in %g, not %g", i, n.Inputs, n.Targets, outputs)
			}
		}
		// fmt.Println("idx", i, "sse", sse, "outputs", outputs, "targets/inputs", n.Targets)
	}

}

// func TestUnitIndex(t *testing.T) {
// 	n := &Network{
// 		NumInputs:       3,
// 		NumOutputs:      7,
// 		NumHiddenLayers: 12,
// 		NumHiddenUnits:  9,
// 	}
// 	n.Init()

// 	// how many times each index has occurred
// 	indexMap := map[int]int{}

// 	// add 2 to account for input and output layers
// 	for layer := 0; layer < n.NumHiddenLayers+2; layer++ {
// 		// the number of units on the current layer
// 		numHiddenUnits := n.NumHiddenUnits
// 		// will be NumInputs/NumOutputs if on input/output layer
// 		if layer == 0 {
// 			numHiddenUnits = n.NumInputs
// 		}
// 		if layer == n.NumHiddenLayers+1 {
// 			numHiddenUnits = n.NumOutputs
// 		}
// 		for unit := 0; unit < numHiddenUnits; unit++ {
// 			ui := n.UnitIndex(layer, unit)
// 			indexMap[ui]++
// 		}
// 	}

// 	for i := 0; i < len(n.Units); i++ {
// 		// each index should only occur once
// 		if indexMap[i] != 1 {
// 			t.Errorf("error: unit index %d occurs %d times, when it should occur 1 time", i, indexMap[i])
// 		}
// 	}
// }

// func TestWeightIndex(t *testing.T) {
// 	n := &Network{
// 		NumInputs:       4,
// 		NumOutputs:      8,
// 		NumHiddenLayers: 9,
// 		NumHiddenUnits:  6,
// 	}
// 	n.Init()

// 	// how many times each index has occurred
// 	indexMap := map[int]int{}

// 	// only add 1 because there are no weights coming from output layer, so we only need to account for input layer
// 	for layer := 0; layer < n.NumHiddenLayers+1; layer++ {
// 		// the number of units on the current layer
// 		numHiddenUnits := n.NumHiddenUnits
// 		// will be NumInputs/NumOutputs if on input/output layer
// 		if layer == 0 {
// 			numHiddenUnits = n.NumInputs
// 		}
// 		for i := 0; i < numHiddenUnits; i++ {
// 			// the number of units on the layer above
// 			numUnitsAbove := n.NumHiddenUnits
// 			// will be NumOutputs if on second to last layer (because it connects to output layer)
// 			if layer == n.NumHiddenLayers {
// 				numUnitsAbove = n.NumOutputs
// 			}
// 			for j := 0; j < numUnitsAbove; j++ {
// 				wi := n.WeightIndex(layer, i, j)
// 				indexMap[wi]++
// 			}
// 		}
// 	}

// 	for i := 0; i < len(n.Weights); i++ {
// 		// each index should only occur once
// 		if indexMap[i] != 1 {
// 			t.Errorf("error: weight index %d occurs %d times, when it should occur 1 time", i, indexMap[i])
// 		}
// 	}
// }

func TestWeightIndex(t *testing.T) {
	n := &Network{
		NumInputs:       4,
		NumOutputs:      8,
		NumHiddenLayers: 9,
		NumHiddenUnits:  6,
	}
	n.Init()

	// how many times each absolute index (relative to the entire network) has occurred
	absoluteIndexMap := map[int]int{}

	// skip input layer because it has no layers connecting to it, only from it
	for li := 1; li < len(n.Layers); li++ {
		layer := n.Layers[li]
		// how many times each local index (relative to the layer) has occurred
		localIndexMap := map[int]int{}
		for i := range layer.Units {
			for h := range n.Layers[li-1].Units {
				wi := layer.WeightIndex(h, i)
				localIndexMap[wi]++
				// need to offset by start for absolute index map
				absoluteIndexMap[wi+layer.WeightsStart]++
			}
		}

		for i := range layer.Weights {
			// each index should only occur once
			if localIndexMap[i] != 1 {
				t.Errorf("error: local weight index %d occurs %d times, when it should occur 1 time", i, localIndexMap[i])
			}
		}
		// fmt.Println(localIndexMap)
	}

	for i := range n.Weights {
		// each index should only occur once
		if absoluteIndexMap[i] != 1 {
			t.Errorf("error: absolute weight index %d occurs %d times, when it should occur 1 time", i, absoluteIndexMap[i])
		}
	}
	// fmt.Println(absoluteIndexMap)
}

func TestForward(t *testing.T) {
	n := &Network{
		LearningRate:    0.1,
		NumInputs:       3,
		NumOutputs:      4,
		NumHiddenLayers: 2,
		NumHiddenUnits:  5,
	}
	n.Init()

	// to test the mechanics of Forward, we set all weights to 0.1
	for wi := range n.Weights {
		n.Weights[wi] = 0.1
	}

	n.Inputs = []float32{6, 4, 7}
	n.Forward()

	want := []Unit{
		// first layer (input layer) should just be the inputs we provided for activation value and net input
		// and should be no error yet because we haven't called Back
		{Act: 6, Net: 6, Err: 0},
		{Act: 4, Net: 4, Err: 0},
		{Act: 7, Net: 7, Err: 0},

		// this is second layer (first hidden layer)
		// net input = sum over i of x[i] * w[i][j]
		// all weights should be 0.1, so = 0.1 * (sum over i of x[i])
		// the sum is just all of the inputs added up, so = 0.1(6+4+7) = 0.1(17) = 1.7
		// activation value = Rectifier(net) = Rectifier(1.7) = 1.7
		// should still be no error because we haven't called Back
		// it is the same for all 5 of them (we said 5 units per hidden layer) because all of the weights are the same
		{Act: 1.7, Net: 1.7, Err: 0},
		{Act: 1.7, Net: 1.7, Err: 0},
		{Act: 1.7, Net: 1.7, Err: 0},
		{Act: 1.7, Net: 1.7, Err: 0},
		{Act: 1.7, Net: 1.7, Err: 0},

		// this is third layer (second hidden layer)
		// net input = sum over i of x[i] * w[i][j]
		// all weights should still be 0.1, so = 0.1 * (sum over i of x[i])
		// the sum is just all of the results from the previous layer added up, so = 0.1(1.7+1.7+1.7+1.7+1.7) = 0.1(5*1.7) = 0.1(8.5) = 0.85
		// activation value = Rectifier(net) = Rectifier(0.85) = 0.85
		// should still be no error because we haven't called Back
		// it is the same for all 5 of them (we said 5 units per hidden layer) because all of the weights are the same
		{Act: 0.85, Net: 0.85, Err: 0},
		{Act: 0.85, Net: 0.85, Err: 0},
		{Act: 0.85, Net: 0.85, Err: 0},
		{Act: 0.85, Net: 0.85, Err: 0},
		{Act: 0.85, Net: 0.85, Err: 0},

		// this is fourth layer (last/output layer)
		// net input = sum over i of x[i] * w[i][j]
		// all weights should still be 0.1, so = 0.1 * (sum over i of x[i])
		// the sum is just all of the results from the previous layer added up, so = 0.1(0.85+0.85+0.85+0.85+0.85) = 0.1(5*0.85) = 0.1(4.25) = 0.425
		// activation value = Logistic(net) = Logistic(0.425) ≈ 0.604679084714
		// should still be no error because we haven't called Back
		// it is the same for all 4 of them (we said 4 units in output layer) because all of the weights are the same
		{Act: 0.604679084714, Net: 0.425, Err: 0},
		{Act: 0.604679084714, Net: 0.425, Err: 0},
		{Act: 0.604679084714, Net: 0.425, Err: 0},
		{Act: 0.604679084714, Net: 0.425, Err: 0},
	}

	for i, unit := range n.Units {
		wantUnit := want[i]
		// the error should be exactly 0 because we aren't doing anything with it, so we don't use about equal for that
		if !(aboutEqual(unit.Act, wantUnit.Act, defTol) && aboutEqual(unit.Net, wantUnit.Net, defTol) && unit.Err == wantUnit.Err) {
			t.Errorf("error: expected at index %d unit %v, but got %v", i, wantUnit, unit)
		}
	}
}

func TestBack(t *testing.T) {
	// we intentionally reuse the same setup as TestForward so that we don't have to compute and test the units after Forward again in TestBack
	// as a result of this, if TestForward fails, TestBack should also fail and should be ignored
	n := &Network{
		LearningRate:    0.1,
		NumInputs:       3,
		NumOutputs:      4,
		NumHiddenLayers: 2,
		NumHiddenUnits:  5,
	}
	n.Init()

	// to test the mechanics of Back, we set all weights to 0.1
	for wi := range n.Weights {
		n.Weights[wi] = 0.1
	}

	n.Inputs = []float32{6, 4, 7}
	n.Forward()

	n.Targets = []float32{9, 14, 8, -3}
	sse := n.Back()

	// just all of the output errors squared added up (these errors are computed below in wantUnits, but because we have sse here, it makes more sense to check it here)
	// = (-8.39532091529)^2 + (-13.3953209153)^2 + (-7.39532091529)^2 + (3.60467908471)^2 = 317.600518438
	wantSSE := float32(317.600518438)

	if !aboutEqual(sse, wantSSE, defTol) {
		t.Errorf("error: sse is %g, but expected %g", sse, wantSSE)
	}

	// NOTE: it makes sense to read these values and comments in reverse order, since the function itself increments backward, and that is how the test values are computed
	wantUnits := []Unit{
		// this is the first layer (input layer)
		// the activation and net input values should still be the same for all of them
		// the error should be sum over j of (error at j * activation function derivative of net input at j * weight between i and j)
		// all of the relevant weights are still 0.1, so it is 0.1 * sum over j of (error at j * activation function derivative of net input at j)
		// there is no i involved in this, so the error will be the same for all of them
		// the net input is the same for all of them, so we can take that out of the sum => = 0.1 * activation function derivative of net input at j * sum over j of (error at j)
		// = 0.1 * RectifierDerivative(1.7) * (0.305750430382+0.305750430382+0.305750430382+0.305750430382+0.305750430382) = 0.152875215191
		{Act: 6, Net: 6, Err: 0.152875215191},
		{Act: 4, Net: 4, Err: 0.152875215191},
		{Act: 7, Net: 7, Err: 0.152875215191},

		// this is the second layer (first hidden layer)
		// the activation and net input values should still be the same for all of them
		// the error should be sum over j of (error at j * activation function derivative of net input at j * weight between i and j)
		// all of the relevant weights are still 0.1, so it is 0.1 * sum over j of (error at j * activation function derivative of net input at j)
		// there is no i involved in this, so the error will be the same for all of them
		// the net input is the same for all of them, so we can take that out of the sum => = 0.1 * activation function derivative of net input at j * sum over j of (error at j)
		// = 0.1 * RectifierDerivative(0.85) * (0.611500860764+0.611500860764+0.611500860764+0.611500860764+0.611500860764) = 0.305750430382
		{Act: 1.7, Net: 1.7, Err: 0.305750430382},
		{Act: 1.7, Net: 1.7, Err: 0.305750430382},
		{Act: 1.7, Net: 1.7, Err: 0.305750430382},
		{Act: 1.7, Net: 1.7, Err: 0.305750430382},
		{Act: 1.7, Net: 1.7, Err: 0.305750430382},

		// this is the third layer (second hidden layer)
		// the activation and net input values should still be the same for all of them
		// the error should be sum over j of (error at j * activation function derivative of net input at j * weight between i and j)
		// all of the weights are still 0.1, so it is 0.1 * sum over j of (error at j * activation function derivative of net input at j)
		// there is no i involved in this, so the error will be the same for all of them
		// the net input is the same for all of them, so we can take that out of the sum => = 0.1 * activation function derivative of net input at j * sum over j of (error at j)
		// = 0.1 * LogisticDerivative(0.425) * (8.39532091529 + 13.3953209153 + 7.39532091529 - 3.60467908471) = 0.611500860764
		{Act: 0.85, Net: 0.85, Err: 0.611500860764},
		{Act: 0.85, Net: 0.85, Err: 0.611500860764},
		{Act: 0.85, Net: 0.85, Err: 0.611500860764},
		{Act: 0.85, Net: 0.85, Err: 0.611500860764},
		{Act: 0.85, Net: 0.85, Err: 0.611500860764},

		// this is the fourth layer (last/output layer)
		// the activation and net input values should still be the same for all of them
		// the error should be Target - Act for each of them
		// = [9-0.604679084714, 14-0.604679084714, 8-0.604679084714, -3-0.6046790847140] = [8.39532091529, 13.3953209153, 7.39532091529, -3.60467908471]
		{Act: 0.604679084714, Net: 0.425, Err: 8.39532091529},
		{Act: 0.604679084714, Net: 0.425, Err: 13.3953209153},
		{Act: 0.604679084714, Net: 0.425, Err: 7.39532091529},
		{Act: 0.604679084714, Net: 0.425, Err: -3.60467908471},
	}

	for i, unit := range n.Units {
		wantUnit := wantUnits[i]
		if !(aboutEqual(unit.Act, wantUnit.Act, defTol) && aboutEqual(unit.Net, wantUnit.Net, defTol) && aboutEqual(unit.Err, wantUnit.Err, defTol)) {
			t.Errorf("error: expected at index %d unit %v, but got %v", i, wantUnit, unit)
		}
	}
	// NOTE: as above with wantUnits, it makes sense to read these values and comments in reverse order, since the function itself increments backward, and that is how the test values are computed
	var wantWeights = []float32{
		// these are the weights connecting from the first layer (input layer) to the second layer (first hidden layer)
		// the delta for each one = -learning rate (which is 0.1) * error for j * activation function derivative of net input at j * activation value at i
		// because net input and error are the same for everything on the second layer,
		// this = -0.1 * -0.305750430382 * RectifierDerivative(1.7) * activation value at i
		// which = 0.0305750430382 * activation value at i
		// actual value = 0.1+delta (because weights start as 0.1) = 0.1 + (0.0305750430382 * activation value at i)
		// due to the way we index weights, we will have everything from i=0 first, then everything from i=1, etc
		// values for i=[0-3]: [0.1+(0.0305750430382*6), 0.1+(0.0305750430382*4), 0.1+(0.0305750430382*7)] = [0.283450258229, 0.222300172153, 0.314025301267]
		// repeated 5 times each because we have 5 j
		0.283450258229, 0.283450258229, 0.283450258229, 0.283450258229, 0.283450258229,
		0.222300172153, 0.222300172153, 0.222300172153, 0.222300172153, 0.222300172153,
		0.314025301267, 0.314025301267, 0.314025301267, 0.314025301267, 0.314025301267,

		// these are the weights connecting from the second layer (first hidden layer) to the third layer (second hidden layer)
		// the delta for each one = -learning rate (which is 0.1) * error for j * activation function derivative of net input at j * activation value at i
		// because net input and error are the same for everything on the third layer and activation value is the same for everything on the second layer,
		// this = -0.1 * -0.611500860764 * RectifierDerivative(0.85) * 1.7
		// which = 0.10395514633
		// actual value = 0.1+delta (because weights start as 0.1) = 0.1 + 0.10395514633 = 0.20395514633
		// repeated 25 times because we have 5 i and 5 j
		0.20395514633, 0.20395514633, 0.20395514633, 0.20395514633, 0.20395514633,
		0.20395514633, 0.20395514633, 0.20395514633, 0.20395514633, 0.20395514633,
		0.20395514633, 0.20395514633, 0.20395514633, 0.20395514633, 0.20395514633,
		0.20395514633, 0.20395514633, 0.20395514633, 0.20395514633, 0.20395514633,
		0.20395514633, 0.20395514633, 0.20395514633, 0.20395514633, 0.20395514633,

		// these are the weights connecting from the third layer (second hidden layer) to the fourth layer (last/output layer)
		// the delta for each one = -learning rate (which is 0.1) * error for j * activation function derivative of net input at j * activation value at i
		// because net input is the same for everything on the fourth layer and activation value is the same for everything on the third layer,
		// this = -0.1 * LogisticDerivative(0.425) * error for j * 0.85
		// which = -0.020318594584 * error for j
		// actual value = 0.1+delta (because weights start as 0.1) = 0.1 - (0.020318594584 * error for j)
		// due to the way we index weights, we will have everything from i=0 first, then everything from i=1, etc
		// values for j=[0-3]: [0.1-(0.020318594584*-8.39532091529), 0.1-(0.020318594584*-13.3953209153), 0.1-(0.020318594584*-7.39532091529), 0.1-(0.020318594584*3.60467908471)]
		// = [0.27058112208, 0.372174095001, 0.250262527496, 0.0267579870724]
		// repeated 5 times because we have 5 i
		0.27058112208, 0.372174095001, 0.250262527496, 0.0267579870724,
		0.27058112208, 0.372174095001, 0.250262527496, 0.0267579870724,
		0.27058112208, 0.372174095001, 0.250262527496, 0.0267579870724,
		0.27058112208, 0.372174095001, 0.250262527496, 0.0267579870724,
		0.27058112208, 0.372174095001, 0.250262527496, 0.0267579870724,
	}
	for i, weight := range n.Weights {
		wantWeight := wantWeights[i]
		if !aboutEqual(weight, wantWeight, defTol) {
			t.Errorf("error: expected at index %d weight %v, but got %v", i, wantWeight, weight)
		}
	}
}

// defTol is a good default tolerance for how much two values can differ to be used with the aboutEqual function
const defTol = 1e-4

// aboutEqual returns whether x is about equal to y with the given tolerance
func aboutEqual(x, y, tol float32) bool {
	diff := x - y
	if diff < 0 {
		diff = -diff
	}
	if diff < tol {
		return true
	}
	return false
}

func BenchmarkNoGoroutines(b *testing.B) {
	n := &Network{
		NumInputs:       1000,
		NumOutputs:      1000,
		NumHiddenLayers: 10,
		NumHiddenUnits:  1000,
		NumGoroutines:   1,
	}
	n.Init()

	for i := 0; i < b.N; i++ {
		n.Forward()
		n.Back()
	}
}

func BenchmarkGoroutines(b *testing.B) {
	n := &Network{
		NumInputs:       1000,
		NumOutputs:      1000,
		NumHiddenLayers: 10,
		NumHiddenUnits:  1000,
	}
	n.Init()

	for i := 0; i < b.N; i++ {
		n.Forward()
		n.Back()
	}
}
