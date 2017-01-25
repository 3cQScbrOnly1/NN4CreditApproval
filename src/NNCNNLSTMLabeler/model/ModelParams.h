#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	vector<Alphabet> vecFeatAlpha;// should be initialized outside
	vector<LookupTable> vecFeatTable;// should be initialized outside
	UniParams _tanh_project;
	LSTMParams _left_lstm_layer;
	LSTMParams _right_lstm_layer;
	BiParams _lstm_concat_project;

	UniParams olayer_linear; // output
public:
	Alphabet labelAlpha; // should be initialized outside
	SoftMaxLoss loss;


public:
	bool initial(HyperParams& opts, AlignedMemoryPool* mem = NULL){

		// some model parameters should be initialized outside
		if (vecFeatAlpha.size() <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.featTypeNum = vecFeatAlpha.size();
		opts.labelSize = labelAlpha.size();
		opts.inputsize = opts.hiddenSize * 3;

		_tanh_project.initial(opts.hiddenSize, opts.featDim * opts.featWindow, mem);
		_left_lstm_layer.initial(opts.rnnHiddenSize, opts.hiddenSize, mem);
		_right_lstm_layer.initial(opts.rnnHiddenSize, opts.hiddenSize, mem);
		_lstm_concat_project.initial(opts.hiddenSize, opts.rnnHiddenSize, opts.rnnHiddenSize, true, mem);
		olayer_linear.initial(opts.labelSize, opts.inputsize, false, mem);
		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		int table_size = vecFeatTable.size();
		for (int idx = 0; idx < table_size; idx++) {
			vecFeatTable[idx].exportAdaParams(ada);
		}
		_tanh_project.exportAdaParams(ada);
		_left_lstm_layer.exportAdaParams(ada);
		_right_lstm_layer.exportAdaParams(ada);
		_lstm_concat_project.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
		//checkgrad.add(&(olayer_linear.b), "olayer_linear.b");
	}

	// will add it later
	void saveModel(){

	}

	void loadModel(const string& inFile){

	}

};

#endif /* SRC_ModelParams_H_ */