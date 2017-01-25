#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:
	int type_size;
	// node instances
	vector<LookupNode> _feat_inputs;
	WindowBuilder _window;
	vector<UniNode> _tanh_hidden;

	LSTMBuilder _left_lstm;
	LSTMBuilder _right_lstm;
	vector<BiNode> _bi_lstm_concat;

	AvgPoolNode _avg_pooling;
	MaxPoolNode _max_pooling;
	MinPoolNode _min_pooling;

	ConcatNode _concat;

	LinearNode _output;
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int feat_type_num){
		_feat_inputs.resize(feat_type_num);
		_window.resize(feat_type_num);
		_tanh_hidden.resize(feat_type_num);
		_left_lstm.resize(feat_type_num);
		_right_lstm.resize(feat_type_num);
		_bi_lstm_concat.resize(feat_type_num);
		_avg_pooling.setParam(feat_type_num);
		_max_pooling.setParam(feat_type_num);
		_min_pooling.setParam(feat_type_num);
		type_size = feat_type_num;
	}

	inline void clear(){
		Graph::clear();
	}

public:
	inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL){
		int feat_type_num = model.vecFeatAlpha.size();
		for (int idx = 0; idx < feat_type_num; idx++) {
			_feat_inputs[idx].setParam(&model.vecFeatTable[idx]);
			_feat_inputs[idx].init(opts.featDim, opts.dropProb, mem);
			_tanh_hidden[idx].setParam(&model._tanh_project);
			_tanh_hidden[idx].init(opts.hiddenSize, opts.dropProb, mem);
			_bi_lstm_concat[idx].setParam(&model._lstm_concat_project);
			_bi_lstm_concat[idx].init(opts.hiddenSize, -1, mem);
		}
		_window.init(opts.featDim, opts.featContext, mem);
		_left_lstm.init(&model._left_lstm_layer, opts.dropProb, true, mem);
		_right_lstm.init(&model._right_lstm_layer, opts.dropProb, true, mem);

		_avg_pooling.init(opts.hiddenSize, -1, mem);
		_min_pooling.init(opts.hiddenSize, -1, mem);
		_max_pooling.init(opts.hiddenSize, -1, mem);
		_concat.init(opts.inputsize, -1, mem);
		_output.setParam(&model.olayer_linear);
		_output.init(opts.labelSize, -1, mem);
	}
	

public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Feature& feature, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation
		// second step: build graph
		//forward
		for (int idx = 0; idx < type_size; idx++) {
			_feat_inputs[idx].forward(this, feature.m_feats[idx]);
		}

		_window.forward(this, getPNodes(_feat_inputs, type_size));
		for (int idx = 0; idx < type_size; idx++) {
			_tanh_hidden[idx].forward(this, &_window._outputs[idx]);
		}

		_left_lstm.forward(this, getPNodes(_tanh_hidden, type_size));
		_right_lstm.forward(this, getPNodes(_tanh_hidden, type_size));
		for (int idx = 0; idx < type_size; idx++) {
			_bi_lstm_concat[idx].forward(this, &_left_lstm._hiddens[idx], &_right_lstm._hiddens[idx]);
		}
		
		_avg_pooling.forward(this, getPNodes(_bi_lstm_concat, type_size));
		_min_pooling.forward(this, getPNodes(_bi_lstm_concat, type_size));
		_max_pooling.forward(this, getPNodes(_bi_lstm_concat, type_size));
		_concat.forward(this, &_avg_pooling, &_max_pooling, &_min_pooling);
		_output.forward(this, &_concat);
	}

};

#endif /* SRC_ComputionGraph_H_ */