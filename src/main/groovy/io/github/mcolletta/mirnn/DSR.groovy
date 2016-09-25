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

import java.util.Random
import groovy.transform.CompileStatic

@CompileStatic
class DSR {

	Random random

	Network network
	int iterations = 150000
	int length = 24	
	int correct = 0
	float success = 0.0f
	float criterion = 0.95f
	float rate = 0.1f
	int symbols

	List<Integer> targets = [2, 4, 7, 8]
	List<Integer> distractors = [3, 5, 6, 9]
	List<Integer> prompts = [0, 1]

	DSR() {
		random = new Random()
	}

	boolean checkOutput(List<Float> output, List<Float> targets) {
		boolean retVal = true
	    output.eachWithIndex { prediction, i ->
	        if (Math.round(prediction) != targets[i]) 
	            retVal = false
	    }
	    return retVal
	}

	Network train() {
		symbols = prompts.size() + targets.size() + distractors.size()
		network = Network.getLSTMg(symbols, symbols, targets.size())

		println "Start learning..."

		int trials = 0
		
		while (trials < iterations && (success < criterion || (trials % 1000) != 0)) {
			List<Integer> sequence = []
			int sequenceLength = length - prompts.size()
			(0..sequenceLength-1).each {
        		sequence << distractors[random.nextInt(distractors.size())]
			}
		
			// Java 8
			def indexes = random.ints(0, targets.size()).limit(prompts.size()).toArray() as List<Integer>
	        def positions = random.ints(0, sequenceLength).distinct().limit(prompts.size()).toArray() as List<Integer>
	        positions.sort()

	        (0..prompts.size()-1).each { i ->
	        	sequence[positions[i]] = targets[indexes[i]]
	        	sequence << prompts[i]
	        }

		    //train sequence		    
		    int distractorsCorrect = 0
		    int targetsCorrect = 0
		    
		    (0..length-1).each { i ->
		        List<Float> input = (0..symbols-1).collect { 0.0f }		        	
		        input[sequence[i]] = 1.0f

		        List<Float> output = (0..targets.size()-1).collect { 0.0f }

		        if (i >= sequenceLength) {
		        	int index = i - sequenceLength
		        	output[indexes[index]] = 1.0f
		        }

		        List<Float> prediction = this.network.activate(input)
		        
		        if (checkOutput(prediction, output)) {
		        	if (i < sequenceLength)
		        		distractorsCorrect++
		        	else
		         		targetsCorrect++
		        } else
		        	network.propagate(output, rate)

		        if ((distractorsCorrect + targetsCorrect) == length) {
		        	correct++
		        }
		    }

		    if (trials % 1000 == 0) 
		        correct = 0
		    trials++
		    int denominator = trials % 1000
		    denominator = (denominator == 0) ? 1000 : denominator
		    success = correct / denominator

		    if (trials % 1000 == 999) {
		    	println "trial $trials with correct $correct and success $success"
		    }
		}

		println "End learning with success: " + success

		return network
	}
}