#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{

	dtype nnRegular; // for optimization
	dtype adaAlpha;  // for optimization
	dtype adaEps; // for optimization

	int hiddenSize;
	dtype dropProb;
	int featDim;
	int featContext;
	int rnnHiddenSize;


	//auto generated
	int inputsize;
	int labelSize;
	int featTypeNum;
	int featWindow;

public:
	HyperParams(){
		bAssigned = false;
	}

public:
	void setRequared(Options& opt){
		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;
		hiddenSize = opt.hiddenSize;
		dropProb = opt.dropProb;
		featDim = opt.featEmbSize;
		featContext = opt.featContext;
		featWindow = opt.featContext * 2 + 1;
		rnnHiddenSize = opt.rnnHiddenSize;
		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}

	bool bValid(){
		return bAssigned;
	}


public:

	void print(){

	}

private:
	bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */