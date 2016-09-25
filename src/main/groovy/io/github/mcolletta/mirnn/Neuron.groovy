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
public class Neuron {

    Connection self_connection = new Connection(this, this, 0.0f)   // by default not connected
    List<Connection> inputs = []
    List<Connection> projected = []
    List<Connection> upstream_prj = []
    List<Connection> gated = []
    HashMap<Neuron, List<Connection>> influenced = [:]              // neurons that have at least a connection gated by this
    HashMap<Connection, Float> trace = [:]                          // elegibility trace
    HashMap<Neuron, HashMap<Connection, Float>> xtrace = [:]        // extended elegibility trace
    HashMap<String, Float> error = [responsibility: 0.0f, projected: 0.0f, gated: 0.0f]
    float state = 0.0f
    float old_state = 0.0f
    float activation = 0.0f
    float derivative = 0.0f
    float bias = ((float)Math.random() * 0.2f - 0.1f)
    Squash squash = new Logistic() 

    Neuron() {}

    // input
    float activate(float input) {
        activation = input
        bias = 0.0f
        return activation
    }

    float activate() {
        old_state = state

        // Eq. 15
        state = self_connection.gain * self_connection.weight * state + bias

        inputs.each { Connection i ->
            state += i.from.activation * i.weight * i.gain
        }

        // Eq. 16
        activation = squash.call(state)

        // f'(s)
        derivative = squash.derivate(state)

        // traces
        inputs.each { Connection i ->
            // elegibility trace - Eq. 17
            trace[i] = self_connection.gain * self_connection.weight * trace[i] + i.gain * i.from.activation

            // extended elegibility trace
            influenced.each { Neuron k, List<Connection> gated_by_this ->
                // the term in parenthesis of Eq. 18
                float influence = (k.self_connection.gater == this) ? k.old_state : 0.0f
                gated_by_this.each { Connection a ->
                    influence += a.weight * a.from.activation
                }
                // Eq. 18
                xtrace[k][i] = (k.self_connection.gain * k.self_connection.weight * xtrace[k][i]) + (derivative * trace[i] * influence)
            }
        }

        //  update gains of gated connections
        gated.each { Connection c ->
          c.gain = activation
        }

        return activation
    }

    // output
    void propagate(float target, float rate) {
        // Eq. 10
        error.responsibility = target - activation
        error.projected = target - activation

        learn(rate)
    }

    void propagate(float rate) {
        float err = 0.0f

        projected.each { Connection connection ->
            Neuron neuron = connection.to
            // Eq. 21
            err += neuron.error.responsibility * connection.gain * connection.weight
        }

        // projected error responsibility
        error.projected = derivative * err

        err = 0.0f
        // error responsibilities from all the connections gated by this neuron
        influenced.each { Neuron neuron, List<Connection> gated_connections ->
            float influence = (neuron.self_connection.gater == this) ? neuron.old_state : 0.0f 
            gated_connections.each { Connection gc ->
                influence += gc.weight * gc.from.activation
            }
            // sum of Eq. 22
            err += neuron.error.responsibility * influence
        }

        // gated error responsibility
        error.gated = derivative * err

        // error responsibility - Eq. 23
        error.responsibility = error.projected + error.gated

        learn(rate)     
    }

    void learn(float rate) {
        inputs.each { Connection i ->
            // Eq. 24
            float gradient = error.projected * trace[i]
            influenced.each { neuron, gated_connections ->
                gradient += neuron.error.responsibility * xtrace[neuron][i]
            }
            i.weight += rate * gradient
        }

        // adjust bias
        bias += rate * error.responsibility
    }

    Connection project(Neuron neuron) {
        float weight = (float)(Math.random() * 0.2f - 0.1f)
        return project(neuron, weight, true)
    }

    Connection project(Neuron neuron, float weight) {
        return project(neuron, weight, true)
    }

    Connection project(Neuron neuron, boolean downstream) {
        float weight = (float)(Math.random() * 0.2f - 0.1f)
        return project(neuron, weight, downstream)
    }

    Connection project(Neuron neuron, float weight, boolean downstream) {
        // self-connection
        if (neuron == this) {
          self_connection.weight = 1
          return self_connection
        }

        Connection connection = projected.find { it.to == neuron }
        if (connection != null) {
            // overwrite weight of existing connection
            connection.weight = weight
        } else {
            // create a new connection
            connection = new Connection(this, neuron, weight)
            if (downstream)
                projected << connection
            else
                upstream_prj << connection
            neuron.inputs << connection
            neuron.trace[connection] = 0.0f
            neuron.xtrace.each { Neuron n, HashMap<Connection, Float> map ->                
                map[connection] = 0.0f
            }
        }
        
        return connection       
    }

    void gate(Connection connection) {
        gated << connection
        Neuron neuron = connection.to

        if (!(neuron in xtrace)) {
            xtrace[neuron] = [:]
            inputs.each { Connection i -> xtrace[neuron][i] = 0.0f }
        }

        if (neuron in influenced)
            influenced[neuron] << connection
        else
            influenced[neuron] = [connection]

        connection.gater = this
    }

}

@CompileStatic
public class Connection {

    Neuron from
    Neuron to 
    float weight
    float gain = 1.0f
    Neuron gater

    Connection(Neuron from, Neuron to, float weight=1.0f) {
        this.from = from
        this.to = to 
        this.weight = weight
    }

    void gate(Neuron gater, float gain) {
        this.gater = gater
        this.gain = gain
    }
}

interface Squash {

    float call(float s)

    float derivate(float s)
}

@CompileStatic
class Logistic implements Squash {

    float call(float x) {
        return 1.0f / (1.0f + Math.exp(-x))
    }

    float derivate(float x) {
        def fx = call(x)
        return fx * (1 - fx)
    }
}
