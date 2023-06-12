package gobp

import (
	"testing"
)

func TestNetwork(t *testing.T) {
	net := NewNetwork(0.1, 1, 2, 2, 2)
	for i := 0; i < 100; i++ {
		if i%2 == 0 {
			net.Units[0].Net = 1
			net.Units[1].Net = 0
			net.Targets[0] = 1
			net.Targets[1] = 0
		} else {
			net.Units[0].Net = 0
			net.Units[1].Net = 1
			net.Targets[0] = 0
			net.Targets[1] = 1
		}
		net.Forward()
		net.Back()
		// log.Println("sse", sse)
		// log.Println("outputs", net.Outputs())
		// log.Println("units", net.Units[4:6])
		// log.Println("weights", net.Weights)
	}
}