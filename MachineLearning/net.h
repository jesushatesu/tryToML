#pragma once
#include <vector>

using namespace std;

class Neuron;

typedef vector<Neuron> Layer;

class net {
public:
	net(const vector<unsigned>& topology);

	//заполняет сеть данными
	void feedForward(const vector<double> &inputVals);

	//алгоритм тренировки сети
	void backProp(const vector<double>& targetVals);

	//вытащить то, что апроксимирует нейронка
	void getResults(vector<double>& resultVals) const;


	double getRecentAverageError() const { return recentAverageError; }

private:
	//обращаться как layer[layerNumber][NeuronNumber]
	vector<Layer> layers;

	double error;
	double recentAverageError;
	static double recentAverageSmoothingFactor;
};