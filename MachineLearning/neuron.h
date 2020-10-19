#pragma once
#include <vector>
#include "net.h"

class Neuron;
typedef vector<Neuron> Layer;

struct Connnection
{
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned index);
	void setOutputVal(double val) { outputVals = val; };
	double getOutputVal() const { return outputVals; };
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double x);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);

private:
	static double randomWeight() { return rand() / double(RAND_MAX); };
	static double activationFunc(double x);
	static double activationFuncDerivative(double x);
	double sumDow(const Layer &nextLayer) const;

	double gradient;
	double outputVals;
	std::vector<Connnection> outputWeights;
	unsigned myIndex;
	static double eta;
	static double alpha;
};