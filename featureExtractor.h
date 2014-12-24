#pragma once


#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <iostream>
#include "DESCache.h"
using namespace cv;

//定义结构体，用来对二次特征提取得到特征进行保存，这里的设计是否合理，还有待进一步的验证，首先先有这样的一个简单框架
//另外还有权重的调整，这里暂时不予考虑
typedef struct _feature
{
	cv::MatND hueHist;
	cv::MatND satHist;
	cv::MatND valHist;
	vector<float> cs_lbpFeature;
	vector<float> cannyFeature;
	cv::MatND horDerHist;
	cv::MatND verDerHist;
	float EHD[5];
}blockFeature;

//提取不同的特征描述，针对不同的特征，特征维度不同，计算过程有所不同
class FeatureExtractor
{
public:

	FeatureExtractor(){};
	~FeatureExtractor(){};

	void initCache();
	const float* getBlockHistogram(float* buf,cv::Mat& img,int nbins);//获得当前窗口内的直方图
    void normalizeBlockHistogram(float* histogram) const;//归一化直方图


	void HSVExtractor(const cv::Mat& src,blockFeature& feature);
	void CS_LBPExtractor(const cv::Mat& gray,blockFeature& feature);
	void CannyExtractor(const cv::Mat& gray,blockFeature& feature);
	void horVerDerExtractor(const cv::Mat& gray,blockFeature& feature);
	void EHDExtarctor(const cv::Mat& gray,blockFeature& feature);

	void computeFeature(const cv::Mat& src,blockFeature &feature);

	//功能单一性，这里仅仅完成二次特征提取工作，至于后续的区分度计算及权重调整，则不需要操心了

private:
	int nbins;//当前单个cell中的直方图维度，lbp为16；H、S、V分别为1，明确
	int winHistogramSize;//单个窗口计算得到描述算子维数
	DESCache cache;

};