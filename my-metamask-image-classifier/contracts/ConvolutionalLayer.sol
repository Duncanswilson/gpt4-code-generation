pragma solidity ^0.8.0;

contract ConvolutionalLayer {
    function applyConvolutionalLayer(
        int[][] memory input,
        int[][][] memory filters,
        int[] memory biases
    ) public pure returns (int[][][] memory) {
        uint numFilters = filters.length;
        uint filterRows = filters[0].length;
        uint filterCols = filters[0][0].length;

        // Assuming input and filters are square matrices and have the same dimensions
        uint inputDim = input.length;
        uint filterDim = filterRows;
        uint outputDim = inputDim - filterDim + 1;

        int[][][] memory output = new int[][][](numFilters);
        for (uint f = 0; f < numFilters; ++f) {
            output[f] = new int[][](outputDim);
            for (uint i = 0; i < outputDim; ++i) {
                output[f][i] = new int[](outputDim);
            }
        }

        for (uint f = 0; f < numFilters; ++f) {
            for (uint x = 0; x < outputDim; ++x) {
                for (uint y = 0; y < outputDim; ++y) {
                    int sum = 0;
                    for (uint i = 0; i < filterDim; ++i) {
                        for (uint j = 0; j < filterDim; ++j) {
                            sum += input[x + i][y + j] * filters[f][i][j];
                        }
                    }
                    output[f][x][y] = sum + biases[f];
                }
            }
        }

        return output;
    }
}
