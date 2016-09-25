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

import io.github.mcolletta.mirnn.*

DSR dsr = new DSR()
dsr.with {
	length = 10
	targets = [2,4]
	distractors = [3,5]
	prompts = [0,1]
	iterations = 25000
	rate = 0.17f
}

dsr.train()