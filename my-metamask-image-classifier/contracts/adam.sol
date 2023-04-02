pragma solidity ^0.8.0;

contract AdamOptimizer {
    struct Param {
        int x;
        int y;
    }

    struct Gradient {
        int grad;
    }

    struct AdamState {
        uint learningRate;
        mapping(Param => int) m1;
        mapping(Param => int) m2;
    }

    function initAdam(uint learningRate) public pure returns (AdamState memory) {
        AdamState memory newState;
        newState.learningRate = learningRate;
        return newState;
    }

    function updateAdam(AdamState storage state, Param memory param, Gradient memory gradient) public {
        int B1 = 9;
        int B2 = 999;
        int E = 10000000;
        int ONE_MINUS_B1 = 1 - B1;
        int ONE_MINUS_B2 = 1 - B2;

        state.m1[param] = state.m1[param] + gradient.grad * ONE_MINUS_B1;
        state.m2[param] = state.m2[param] + gradient.grad * gradient.grad * ONE_MINUS_B2;

        int m1_hat = state.m1[param] / ONE_MINUS_B1;
        int m2_hat = state.m2[param] / ONE_MINUS_B2;

        int paramUpdate = int(state.learningRate) * m1_hat / (sqrt(m2_hat) + E);
        param.x = param.x - paramUpdate;
    }

    function sqrt(int x) private pure returns (int) {
        int z = (x + 1) / 2;
        int y = x;

        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
        return y;
    }
}
