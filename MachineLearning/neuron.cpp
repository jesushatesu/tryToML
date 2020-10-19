#include "neuron.h"
#include <cmath>


double Neuron::alpha = 0.5; // momentum тобишь динамика
double Neuron::eta = 0.15; //net learning rate

Neuron::Neuron(unsigned numOutputs, unsigned index)
{
	for (unsigned c = 0; c < numOutputs; c++)
	{
		outputWeights.push_back(Connnection());
		outputWeights.back().weight = randomWeight();
	}

	myIndex = index;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.outputWeights[myIndex].deltaWeight;
		double newDeltaWeight = eta * neuron.getOutputVal() * gradient + alpha * oldDeltaWeight;

		neuron.outputWeights[myIndex].deltaWeight = newDeltaWeight;
		neuron.outputWeights[myIndex].weight += newDeltaWeight;

		double a = newDeltaWeight;
	}
}

double Neuron::sumDow(const Layer &nextLayer) const
{
	double sum = 0.0;

	for (unsigned n = 0; n < nextLayer.size() - 1; n++)
	{
		sum += outputWeights[n].weight * nextLayer[n].gradient;
	}

	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDow(nextLayer);
	gradient = dow * activationFuncDerivative(outputVals);
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - outputVals;
	gradient = delta * activationFuncDerivative(outputVals);
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		sum += prevLayer[n].getOutputVal() * prevLayer[n].outputWeights[myIndex].weight;
	}
	
	outputVals = activationFunc(sum);
}

double Neuron::activationFunc(double x)
{
	return tanh(x);
}

double Neuron::activationFuncDerivative(double x)
{
	return 1 - x * x;
}




