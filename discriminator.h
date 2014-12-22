#pragma once
#include "featureExtractor.h"

class Discriminator
{
private:
	double weights[8];
	double distance[8];//用于保存一个feature与另一个feature的distance
	blockFeature current;
	cv::MatND targetLBP;
	cv::MatND currentLBP;
	cv::MatND targetCanny;
	cv::MatND currentCanny;
	cv::MatND targetEHD;
	cv::MatND currentEHD;

	int count;//统计连续计算feature距离个数
public:
	Discriminator();

	void setCurrentFeature(blockFeature& feature);//设定当前feature，之后只需要依次与目标feature进行比较即可
	void computeDistance(blockFeature& targe);//计算两feature区分度
	void clearDistance();//对距离信息进行清空
	void computeDistanceHold(blockFeature& target);//连续计算当前feature与targetFeature平均距离
	double distinguish();

	void getDistance(double outputDistance[]);//将计算距离输出，用于其他操作
	void setWeights(double inputWeights[]);//根据修正结果权重进行调节
};




