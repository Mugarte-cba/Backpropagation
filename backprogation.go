package main

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"fmt"
	"math"
	"math/rand"
	"errors"
	"strconv"
	"time"
)
func sigmoid(x float64) float64 {
        return 1.0 / (1.0 + math.Exp(-x))
}
func sigmoidPrime(x float64) float64 {
    return sigmoid(x) * (1.0 - sigmoid(x))
}
type neuralNet struct {
        config  neuralNetConfig
        wHidden *mat.Dense
        bHidden *mat.Dense
        wOut    *mat.Dense
        bOut    *mat.Dense
}
type neuralNetConfig struct {
        inputNeurons  int
        outputNeurons int
        hiddenNeurons int
        numEpochs     int
        learningRate  float64
}
func newNetwork(config neuralNetConfig) *neuralNet {
        return &neuralNet{config: config}
}
func (nn *neuralNet) train(x, y *mat.Dense) error {

    randSource := rand.NewSource(time.Now().UnixNano())
    randGen := rand.New(randSource)

    wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
    bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
    wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
    bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

    wHiddenRaw := wHidden.RawMatrix().Data
    bHiddenRaw := bHidden.RawMatrix().Data
    wOutRaw := wOut.RawMatrix().Data
    bOutRaw := bOut.RawMatrix().Data

    for _, param := range [][]float64{
        wHiddenRaw,
        bHiddenRaw,
        wOutRaw,
        bOutRaw,
    } {
        for i := range param {
            param[i] = randGen.Float64()
        }
    }
    output := new(mat.Dense)

    if err := nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
        return err
    }

    nn.wHidden = wHidden
    nn.bHidden = bHidden
    nn.wOut = wOut
    nn.bOut = bOut

    return nil
}
func (nn *neuralNet) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {

    for i := 0; i < nn.config.numEpochs; i++ {

        hiddenLayerInput := new(mat.Dense)
        hiddenLayerInput.Mul(x, wHidden)
        addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
        hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

        hiddenLayerActivations := new(mat.Dense)
        applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
        hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

        outputLayerInput := new(mat.Dense)
        outputLayerInput.Mul(hiddenLayerActivations, wOut)
        addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
        outputLayerInput.Apply(addBOut, outputLayerInput)
        output.Apply(applySigmoid, outputLayerInput)

        networkError := new(mat.Dense)
        networkError.Sub(y, output)

        slopeOutputLayer := new(mat.Dense)
        applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
        slopeOutputLayer.Apply(applySigmoidPrime, output)
        slopeHiddenLayer := new(mat.Dense)
        slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

        dOutput := new(mat.Dense)
        dOutput.MulElem(networkError, slopeOutputLayer)
        errorAtHiddenLayer := new(mat.Dense)
        errorAtHiddenLayer.Mul(dOutput, wOut.T())

        dHiddenLayer := new(mat.Dense)
        dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

        wOutAdj := new(mat.Dense)
        wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
        wOutAdj.Scale(nn.config.learningRate, wOutAdj)
        wOut.Add(wOut, wOutAdj)

        bOutAdj, err := sumAlongAxis(0, dOutput)
        if err != nil {
            return err
        }
        bOutAdj.Scale(nn.config.learningRate, bOutAdj)
        bOut.Add(bOut, bOutAdj)

        wHiddenAdj := new(mat.Dense)
        wHiddenAdj.Mul(x.T(), dHiddenLayer)
        wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
        wHidden.Add(wHidden, wHiddenAdj)

        bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
        if err != nil {
            return err
        }
        bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
        bHidden.Add(bHidden, bHiddenAdj)
    }

    return nil
}
func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {

    if nn.wHidden == nil || nn.wOut == nil {
        return nil, errors.New("the supplied weights are empty")
    }
    if nn.bHidden == nil || nn.bOut == nil {
        return nil, errors.New("the supplied biases are empty")
    }

    output := new(mat.Dense)

    hiddenLayerInput := new(mat.Dense)
    hiddenLayerInput.Mul(x, nn.wHidden)
    addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
    hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

    hiddenLayerActivations := new(mat.Dense)
    applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
    hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

    outputLayerInput := new(mat.Dense)
    outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
    addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
    outputLayerInput.Apply(addBOut, outputLayerInput)
    output.Apply(applySigmoid, outputLayerInput)

    return output, nil
}