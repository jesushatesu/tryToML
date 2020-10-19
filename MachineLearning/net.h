#pragma once
#include <vector>

using namespace std;

class Neuron;

typedef vector<Neuron> Layer;

class net {
public:
	net(const vector<unsigned>& topology);

	//��������� ���� �������
	void feedForward(const vector<double> &inputVals);

	//�������� ���������� ����
	void backProp(const vector<double>& targetVals);

	//�������� ��, ��� ������������� ��������
	void getResults(vector<double>& resultVals) const;


	double getRecentAverageError() const { return recentAverageError; }

private:
	//���������� ��� layer[layerNumber][NeuronNumber]
	vector<Layer> layers;

	double error;
	double recentAverageError;
	static double recentAverageSmoothingFactor;
};