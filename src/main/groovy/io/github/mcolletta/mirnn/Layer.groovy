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
class Layer {

	List<Neuron> neurons = []

	Layer(int size, float bias=0.0f) {
		(1..size).each {
			Neuron neuron = new Neuron()
			if (bias != 0.0f)
				neuron.bias = bias
			neurons << neuron
		}
	}

	List<Float> activate(List<Float> input) {
		assert input.size() == neurons.size()
		List<Float> activations = []
		neurons.eachWithIndex { Neuron neuron, int i ->
			activations << (neuron.activate(input[i]))
		}
		return activations
	}

	List<Float> activate() {
		List<Float> activations = []
		neurons.eachWithIndex { Neuron neuron, int i ->
			activations << (neuron.activate())
		}
		return activations
	}

	void propagate(List<Float> target, float rate) {
		assert target.size() == neurons.size()
		for (int i = neurons.size() - 1; i >= 0; i--) {
	        Neuron neuron = neurons[i]
	        neuron.propagate(target[i], rate)
	    }
	}

	void propagate(float rate) {
		neurons.reverseEach { Neuron neuron -> neuron.propagate(rate)}
	}

	LayerConnection project(Layer layer, boolean downstream=true) {
		LayerConnectionType lctype = LayerConnectionType.ALL_TO_ALL
		project(layer, lctype, downstream)
	}

	LayerConnection project(Layer layer, LayerConnectionType lctype, boolean downstream) {
		def layerConnection = new LayerConnection(this, layer)
		layerConnection.downstream = downstream
		if (layer == this) // self connection
			lctype = LayerConnectionType.ONE_TO_ONE
		layerConnection.lctype = lctype
		switch (lctype) {
			case { it == LayerConnectionType.ALL_TO_ALL || it == LayerConnectionType.ALL_TO_ELSE }:
				neurons.each { Neuron from ->
					layer.neurons.each { Neuron to ->
						if (!(lctype == LayerConnectionType.ALL_TO_ELSE && from == to))
							layerConnection.connections << from.project(to, downstream)
					}
				}
				break
			case LayerConnectionType.ONE_TO_ONE:
				assert neurons.size() == layer.neurons.size()
				neurons.eachWithIndex { Neuron from, int i ->
					Neuron to = layer.neurons[i]
					layerConnection.connections << from.project(to, downstream)
				}
				break
			default:
				break
		}
		return layerConnection
	}

	void gate(LayerConnection layerConnection, LayerGateType lgtype=LayerGateType.ONE_TO_ONE) {
		List<Connection> connections = layerConnection.connections
		switch (lgtype) {
			case LayerGateType.INPUT:
				Layer toLayer = layerConnection.to
				assert toLayer.neurons.size() == neurons.size()
				toLayer.neurons.eachWithIndex { Neuron  neuron, int i ->
					Neuron gater = neurons[i]
					neuron.inputs.each { Connection gated ->
						if (gated in connections)
							gater.gate(gated)
					}
				}
				break
			case LayerGateType.OUTPUT:
				Layer fromLayer = layerConnection.from
				assert fromLayer.neurons.size() == neurons.size()
				fromLayer.neurons.eachWithIndex { Neuron neuron, int i ->
					Neuron gater = neurons[i]
					neuron.projected.each { Connection gated ->
						if (gated in connections)
							gater.gate(gated)
					}
				}
				break
			case LayerGateType.ONE_TO_ONE:
				assert connections.size() == neurons.size()
				connections.eachWithIndex { Connection gated, int i ->
					Neuron gater = neurons[i]
					gater.gate(gated)
				}
				break
			default:
				break
		}
		layerConnection.gater = this
	}

}

@CompileStatic
class LayerConnection {
	Layer from
	Layer to 
	List<Connection> connections = []
	Layer gater
	LayerConnectionType lctype = LayerConnectionType.ALL_TO_ALL

	boolean downstream = true

	LayerConnection(Layer from, Layer to)  {
		this.from = from
		this.to = to
	}
}

enum LayerConnectionType {
	ALL_TO_ALL,
	ONE_TO_ONE,
	ALL_TO_ELSE
}

enum LayerGateType {
	INPUT,
	OUTPUT,
	ONE_TO_ONE
}