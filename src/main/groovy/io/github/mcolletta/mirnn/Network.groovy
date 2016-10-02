/*
 * Copyright (C) 2016 Mirco Colletta
 *
 * This file is part of MiRNN.
 *
 */

/**
 * @author Mirco Colletta
 */

package io.github.mcolletta.mirnn

import groovy.transform.CompileStatic

@CompileStatic
class Network {

	Layer input
	List<Layer> hidden
	Layer output

	Network() {}

	Network(Layer input, List<Layer> hidden, Layer output) {
		this.input = input
		this.hidden = hidden
		this.output = output
	}

	void setLayers(Layer input, List<Layer> hidden, Layer output) {
		this.input = input
		this.hidden = hidden
		this.output = output
	}

	List<Float> activate(List<Float> inp) {
		input.activate(inp)
		for(Layer l : hidden) {
			l.activate()
		}
		return output.activate()
	}

	void propagate(List<Float> target, float rate=0.1f) {
		output.propagate(target, rate)
		for (int i = hidden.size() - 1; i >= 0; i--) {
			Layer l = hidden[i]
			l.propagate(rate)
		}
	}

	static Network getLSTMg(int input, int blocks, int output) {
		Network network = new Network()

		Layer inputLayer = new Layer(input)
		Layer inputGate = new Layer(blocks, 1.0f)
		Layer forgetGate = new Layer(blocks, 1.0f)
		Layer memoryCell = new Layer(blocks)
		Layer outputGate = new Layer(blocks, 1.0f)
		Layer outputLayer = new Layer(output)

		LayerConnection input_connection = inputLayer.project(memoryCell)
		inputLayer.project(inputGate)
		inputLayer.project(forgetGate)
		inputLayer.project(outputGate)

		LayerConnection output_connection =  memoryCell.project(outputLayer)

		LayerConnection self_connection = memoryCell.project(memoryCell)

		// peepholes
		memoryCell.project(inputGate, false)
		memoryCell.project(forgetGate, false)
		memoryCell.project(outputGate)

		inputGate.gate(input_connection, LayerGateType.INPUT)
		forgetGate.gate(self_connection, LayerGateType.ONE_TO_ONE)
		outputGate.gate(output_connection, LayerGateType.OUTPUT)

		// input to output direct connection
		inputLayer.project(outputLayer)

		network.setLayers(inputLayer, [inputGate, forgetGate, memoryCell, outputGate], outputLayer)

		return network
	}

}
