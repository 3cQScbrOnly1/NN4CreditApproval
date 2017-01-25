#ifndef _CONLL_READER_
#define _CONLL_READER_

#include "Reader.h"
#include "N3L.h"
#include <sstream>

using namespace std;
/*
 this class reads conll-format data (10 columns, no srl-info)
 */
class InstanceReader : public Reader {
public:
	InstanceReader() {
	}
	~InstanceReader() {
	}

	Instance *getNext() {
		m_instance.clear();
		string strLine;
		if (!my_getline(m_inf, strLine))
			return NULL;
		if (strLine.empty())
			return NULL;


		vector<string> vecInfo;
		split_bychar(strLine, vecInfo, ',');
		m_instance.allocate(vecInfo.size() - 1);
		m_instance.m_label = vecInfo[vecInfo.size() - 1];

		m_instance.m_feats.assign(vecInfo.begin(), vecInfo.end() - 1);
		return &m_instance;
	}
};

#endif

