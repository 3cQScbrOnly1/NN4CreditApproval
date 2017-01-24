#ifndef _INSTANCE_H_
#define _INSTANCE_H_

#include <iostream>
using namespace std;

class Instance
{
public:
	void clear()
	{
		m_feats.clear();
		m_label.clear();
	}

	void evaluate(const string& predict_label, Metric& eval) const
	{
		if (predict_label == m_label)
			eval.correct_label_count++;
		eval.overall_label_count++;
	}

	void copyValuesFrom(const Instance& anInstance)
	{
		allocate(anInstance.size());
		m_label = anInstance.m_label;
		m_feats = anInstance.m_feats;
	}

	void assignLabel(const string& resulted_label) {
		m_label = resulted_label;
	}

	int size() const {
		return m_feats.size();
	}

	void allocate(int length)
	{
		clear();
		m_feats.resize(length);
	}
public:
	vector<string> m_feats;
	string m_label;
};

#endif /*_INSTANCE_H_*/
