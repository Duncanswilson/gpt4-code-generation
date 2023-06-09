{-# LANGUAGE FlexibleContexts #-}

module NeuralNetwork where

import Numeric.LinearAlgebra

-- Define the activation function and its derivative
sigmoid :: Double -> Double
sigmoid x = 1 / (1 + exp (-x))

sigmoid' :: Double -> Double
sigmoid' x = sigmoid x * (1 - sigmoid x)

-- Define the neural network structure
type Weights = Matrix Double
type Biases = Vector Double
type Network = ([Weights], [Biases])

-- Initialize the network with random weights and biases
initializeNetwork :: [Int] -> IO Network
initializeNetwork layers = do
    ws <- mapM (\(x, y) -> randn x y) (zip layers (tail layers))
    bs <- mapM (\x -> randn x 1) (tail layers)
    return (ws, fmap flatten bs)

-- Feedforward through the network
feedforward :: Network -> Vector Double -> Vector Double
feedforward (weights, biases) input = foldl (\acc (w, b) -> cmap sigmoid (w #> acc + b)) input (zip weights biases)

-- Train the network using the backpropagation algorithm
trainNetwork :: Network -> [(Vector Double, Vector Double)] -> Double -> Int -> IO Network
trainNetwork network _ _ 0 = return network
trainNetwork network trainingData learningRate epochs = do
    newNetwork <- updateNetwork network trainingData learningRate
    trainNetwork newNetwork trainingData learningRate (epochs - 1)

-- Update the network using the gradient descent
updateNetwork :: Network -> [(Vector Double, Vector Double)] -> Double -> Network
updateNetwork (weights, biases) trainingData eta =
    let
        num = fromIntegral $ length trainingData
        (weightUpdates, biasUpdates) = unzip $ fmap (\(x, y) -> backpropagate (weights, biases) (x, y)) trainingData
        avgWeightUpdates = fmap (/ num) . sum <$> transpose weightUpdates
        avgBiasUpdates = fmap (/ num) . sum <$> (transpose $ fmap asColumn <$> biasUpdates)
        newWeights = zipWith (-) weights avgWeightUpdates
        newBiases = zipWith (-) biases (fmap flatten <$> avgBiasUpdates)
    in (newWeights, newBiases)

-- Backpropagation algorithm
backpropagate :: Network -> (Vector Double, Vector Double) -> ([Weights], [Biases])
backpropagate (weights, biases) (x, y) = (weightUpdates, biasUpdates)
    where
        weightedInputs = scanl (\acc (w, b) -> w #> acc + b) x (zip weights biases)
        activations = fmap (cmap sigmoid) weightedInputs
        delta = (last activations - y) * cmap sigmoid' (last weightedInputs)
        deltas = reverse . snd $ foldr (\(w, a, z) (d, acc) -> let newD = (tr w #> d) * cmap sigmoid' z in (newD, newD : acc)) (delta, []) (zip3 (init weights) (init activations) (init weightedInputs))
        weightUpdates = zipWith outer deltas (init activations)
        biasUpdates = fmap flatten $ fmap asColumn deltas
