#include "net.h"
#include "neuron.h"
#include <cassert>

double net::recentAverageSmoothingFactor = 100.0;

net::net(const vector<unsigned>& topology)
{
	unsigned numLayers = topology.size();

	for (unsigned layersNum = 0; layersNum < numLayers; layersNum++)
	{
		layers.push_back(Layer());
		const unsigned numOutputs = layersNum == topology.size() - 1 ? 0 : topology[layersNum + 1];
		for (unsigned neuronNum = 0; neuronNum <= topology[layersNum]; neuronNum++)
		{
			layers.back().push_back(Neuron(numOutputs, neuronNum));
		}
		layers.back().back().setOutputVal(1.0);
	}

	recentAverageError = 1;
}

void net::feedForward(const vector<double>& inputVals)
{
	assert(inputVals.size() == layers[0].size() - 1);

	for (unsigned i = 0; i < inputVals.size(); i++)
	{
		layers[0][i].setOutputVal(inputVals[i]);
	}

	for (unsigned layerNum = 1; layerNum < layers.size(); layerNum++)
	{
		Layer& prevLayer = layers[layerNum - 1];

		for (unsigned n = 0; n < layers[layerNum].size(); n++)
		{
			layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void net::backProp(const vector<double>& targetVals)
{
	//вычисление ошибки вычисления нейронки
	Layer& outoutLayer = layers.back();
	error = 0.0;

	for (unsigned n = 0; n < outoutLayer.size(); n++)
	{
		double delta = targetVals[n] - outoutLayer[n].getOutputVal();
		error += delta * delta;
	}
	error /= outoutLayer.size() - 1;
	error = sqrt(error);

	recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);


	//вычисляем градиент для каждого слоя
	//вычисляем градиент для последнего слоя
	for (unsigned n = 0; n < outoutLayer.size(); n++)
	{
		outoutLayer[n].calcOutputGradients(targetVals[n]);
	}

	//вычисляем градиент для скрытого слоя
	for (unsigned layerNum = layers.size() - 2; layerNum > 0; layerNum--)
	{
		Layer& hiddenLayer = layers[layerNum];
		Layer& nextLayer = layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); n++)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	//обновляем веса каждого нейрона
	for (unsigned layerNum = layers.size() - 2; layerNum > 0; layerNum--)
	{
		Layer& layer = layers[layerNum];
		Layer& prevLayer = layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size(); n++)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void net::getResults(vector<double>& resultVals) const
{
	resultVals.clear();

	for (unsigned n = 0; n < layers.back().size(); n ++)
	{
		resultVals.push_back(layers.back()[n].getOutputVal());
	}
}

