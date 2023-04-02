// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "https://github.com/abdk-consulting/abdk-libraries-solidity/blob/master/ABDKMath64x64.sol";

contract SimplePerceptron {
    using ABDKMath64x64 for int128;

    int128[] public weights;
    int128 public bias;
    int128 public learningRate;

    constructor(uint256 numInputs, int128 _learningRate) {
        weights = new int128[](numInputs);
        for (uint256 i = 0; i < numInputs; i++) {
            weights[i] = ABDKMath64x64.fromUInt(1);
        }
        bias = ABDKMath64x64.fromUInt(1);
        learningRate = _learningRate;
    }

    function predict(int128[] memory inputs) public view returns (int128) {
        require(inputs.length == weights.length, "Invalid input length");

        int128 sum = bias;
        for (uint256 i = 0; i < inputs.length; i++) {
            sum = sum.add(inputs[i].mul(weights[i]));
        }

        return ABDKMath64x64.sigmoid(sum);
    }

    function train(int128[] memory inputs, int128 target) public {
        int128 prediction = predict(inputs);
        int128 error = target.sub(prediction);

        // Update weights and bias
        for (uint256 i = 0; i < weights.length; i++) {
            int128 gradient = inputs[i].mul(error).mul(prediction).mul(ABDKMath64x64.fromUInt(1).sub(prediction));
            int128 weightDelta = learningRate.mul(gradient);
            weights[i] = weights[i].add(weightDelta);
        }

        int128 biasGradient = error.mul(prediction).mul(ABDKMath64x64.fromUInt(1).sub(prediction));
        int128 biasDelta = learningRate.mul(biasGradient);
        bias = bias.add(biasDelta);
    }
}
