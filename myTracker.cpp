#include "myTracker.h"
//将检测得到pedArea存储到tracker中，这里不能直接使用引用吧，
//假设是使用同一内存空间，则后续的改变，将对其造成影响。，貌似没有问题，等出现问题再进行修改吧

Tracker::Tracker()
{
	//完成对kalman滤波器的初始化操作，用于之后的tracklet的预测过程
	KF = cv::KalmanFilter(stateNum, measureNum, 0);
	state = cv::Mat(stateNum, 1, CV_32F);//滤波器状态矩阵
	processNoise = cv::Mat(stateNum, 1, CV_32F);//滤波器处理噪声
	measurement = cv::Mat::zeros(measureNum, 1, CV_32F);//滤波器测量矩阵
	KF.transitionMatrix = *( cv::Mat_<float>(4, 4) << 1,0,1,0,0,1,0,1,0,0,1,0,0,0,0,1 );//转移矩阵

	setIdentity(KF.measurementMatrix);//测量矩阵
	setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));
	setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, cv::Scalar::all(1));
	randn(KF.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));

	//对二次特征提取模块进行初始化操作
	extractor = FeatureExtractor();
	extractor.initCache();
	lockedPedArea = NULL;
	trackerletHead = NULL;
}


void Tracker::setLoackedPedArea(LockedArea* result)
{
	lockedPedArea = result;//直接指向同一内存空间就好了，暂时先这样处理，后续可能会要进行改变
}
bool Tracker::update(cv::Mat &sourceImage)
{
	if(trackerletHead == NULL)
	{
		if(lockedPedArea == NULL)
			return false;
		else
		{
			//根据lockedPedArea产生新的tracklet
			int topLeftX = lockedPedArea->topLeftX;
			int topLeftY = lockedPedArea->topLeftY;
			int width = lockedPedArea->width;
			int height = lockedPedArea->height;
			int letWidth = width * 0.5;
			int letHeight = height * 0.25;
			int letTopLeftX = topLeftX + width * 0.25;
			int letTopLeftY = topLeftY + height * 0.25;
			cv::Mat subImage = sourceImage(cv::Rect(letTopLeftX,letTopLeftY,letWidth,letHeight));
			cv::rectangle(sourceImage,cv::Rect(letTopLeftX,letTopLeftY,letWidth,letHeight),cv::Scalar(255,0,0),2);
			//blockFeature target;
			//extractor.computeFeature(subImage,target);
			//Trackerlet* trackerlet = new Trackerlet();
			//trackerlet->topLeftX = letTopLeftX;
			//trackerlet->topLeftY = letTopLeftY;
			//trackerlet->width = letWidth;
			//trackerlet->Height = letHeight;
			//trackerlet->next = NULL;
			//trackerlet->trackerletID = 0;//这个ID暂时没有什么意义，先临时放在这里
			return true;
		}
	}
	else
	{
		return true;
	}
}