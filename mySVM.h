#pragma once
//不调用代码，用手动来实现，方便调试，知道是哪里耗费的时间
#include "SymmetryProcess.h"
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>

#include <time.h>

class MySVM : public CvSVM
{
public:
	//获得SVM的决策函数中的alpha数组
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}

	//获得SVM的决策函数中的rho参数,即偏移量
	float get_rho()
	{
		return this->decision_func->rho;
	}
};

class SVMDetector
{
public:
	SVMDetector();
	~SVMDetector(){};//完成必要的清除工作
	void loadImage(const char* filename);//加载待检测的图像
	void loadImage(cv::Mat& image);//重载另外一种加载图像方式
	void saveImage(const char* filename);//保存目标图像

	int computeDetectorVector();
	void saveDetectorVector(const char* filename);
	void loadDetectorVector(const char* filename);

	void computeDescriptor(cv::Mat& tmpImage,std::vector<float>& descriptor);
	bool predict(std::vector<float>& samples,int var_count);//利用给定的分类器对指定samples进行预测，

	void initSymmetryParam();//对基于对称性行人检测的参数进行初始化，
	void initSymmetryParam(float ax,float ay,float u0,float v0,float f,float h);
	void detectBaseOnSymmetry(cv::Mat& sourceImage);//给定参数为单通道灰度图像，

	int detectOnVideo(const char* filename);//对视频流进行检测，

	LockedArea* getResultRect();

private:
	
	SymmetryProcess sp;
	cv::Mat sourceImage;//待检测的图像信息
	LockedArea* lockedPedArea;
	MySVM svm_classifier;//声明svm分类器，在滑动窗口及候选区域验证过程中均使用该分类器
	cv::HOGDescriptor hog;

	std::vector<float> myDetector;//真正参与预测的
	int detectorDim;//记录detector的维度

	cv::vector<cv::Rect> resultRect;
};
//这个流程暂且放一放，重点是完成论文，