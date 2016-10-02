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
		for(int i = 0; i < size; i++) {
			Neuron neuron = new Neuron()
			if (bias != 0.0f)
				neuron.bias = bias
			neurons << neuron
		}
	}

	List<Float> activate(List<Float> input) {
		assert input.size() == neurons.size()
		List<Float> activations = []
		for(int i = 0; i < neurons.size(); i++) {
			Neuron neuron = neurons[i]
			activations << (neuron.activate(input[i]))
		}
		return activations
	}

	List<Float> activate() {
		List<Float> activations = []
		for(int i = 0; i < neurons.size(); i++) {
			Neuron neuron = neurons[i]
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
		for (int i = neurons.size() - 1; i >= 0; i--) {
	        Neuron neuron = neurons[i]
			neuron.propagate(rate)
		}
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
				for(Neuron from : neurons) {
					for(Neuron to : layer.neurons) {
						if (!(lctype == LayerConnectionType.ALL_TO_ELSE && from == to))
							layerConnection.connections << from.project(to, downstream)
					}
				}
				break
			case LayerConnectionType.ONE_TO_ONE:
				assert neurons.size() == layer.neurons.size()
				for(int i = 0; i < neurons.size(); i++) {
					Neuron from = neurons[i]
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
				for(int i = 0; i < toLayer.neurons.size(); i++) {
					Neuron neuron = toLayer.neurons[i]
					Neuron gater = neurons[i]
					for(Connection gated : neuron.inputs) {
						if (gated in connections)
							gater.gate(gated)
					}
				}
				break
			case LayerGateType.OUTPUT:
				Layer fromLayer = layerConnection.from
				assert fromLayer.neurons.size() == neurons.size()
				for(int i = 0; i < fromLayer.neurons.size(); i++) {
					Neuron neuron = fromLayer.neurons[i]
					Neuron gater = neurons[i]
					for(Connection gated : neuron.projected) {
						if (gated in connections)
							gater.gate(gated)
					}
				}
				break
			case LayerGateType.ONE_TO_ONE:
				assert connections.size() == neurons.size()
				for(int i = 0; i < connections.size(); i++) {
					 Connection gated = connections[i]
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