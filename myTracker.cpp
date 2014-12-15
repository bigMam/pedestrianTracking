#include "myTracker.h"
//将检测得到pedArea存储到tracker中，这里不能直接使用引用吧，
//假设是使用同一内存空间，则后续的改变，将对其造成影响。，貌似没有问题，等出现问题再进行修改吧

Tracker::Tracker()
{
	//完成对kalman滤波器的初始化操作，用于之后的tracklet的预测过程
	stateNum = 8;
	measureNum = 4;

	KF = cv::KalmanFilter(stateNum, measureNum, 0);
	state = cv::Mat(stateNum, 1, CV_32F);//滤波器状态矩阵
	processNoise = cv::Mat(stateNum, 1, CV_32F);//滤波器处理噪声
	measurement = cv::Mat::zeros(measureNum, 1, CV_32F);//滤波器测量矩阵
	KF.transitionMatrix = *( Mat_<float>(8, 8) << 1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,
		0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,
		0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,
		0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1);//转移矩阵

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
//更新过程的思路整理，
//一是直接根据行人检测矩形框进行tracklet提取
//二是根据kalman预测结果进行tracklet提取，并与之前tracklet进行比较
//如果差别在可接受范围内，则进行更新，否则认为更新失败，需要想detector发送检测请求
//返回值为isRequset,表示当前更新后是否需要进行检测，更新失败则需要进行检测，
bool Tracker::update(cv::Mat &sourceImage,bool haveRectBoxing)
{
	if(haveRectBoxing)//表示当前有新鲜出炉的行人检测矩形框，不需要进行预测过程？有待商榷
	{
		if(lockedPedArea == NULL)//检测，但是没有检测到行人。虽然没有检测到行人但是依然不能够使用预测方法，
			//原因：之前的预测结果已经出现问题当前同样无法进行预测过程
			return true;
		else//有检测行人可以根据检测行人对tracklet进行更新过程，这里如何对tracklet进行管理还没有明确的思路，暂时先仅对tracklet进行管理
			//先写出一个流程出来，
		{
			//根据lockedPedArea产生新的tracklet
			int topLeftX = lockedPedArea->topLeftX;
			int topLeftY = lockedPedArea->topLeftY;
			int width = lockedPedArea->width;
			int height = lockedPedArea->height;

			int letWidth = width * 0.4;
			int letHeight = height * 0.18;
			int letTopLeftX = topLeftX + width * 0.3;
			int letTopLeftY = topLeftY + height * 0.25;
			cv::Mat subImage = sourceImage(cv::Rect(letTopLeftX,letTopLeftY,letWidth,letHeight));
			cv::rectangle(sourceImage,cv::Rect(letTopLeftX,letTopLeftY,letWidth,letHeight),cv::Scalar(255,0,0),2);

			blockFeature target;
			extractor.computeFeature(subImage,target);

			Trackerlet* trackerlet = new Trackerlet();
			trackerlet->topLeftX = letTopLeftX;
			trackerlet->topLeftY = letTopLeftY;
			trackerlet->width = letWidth;
			trackerlet->Height = letHeight;
			trackerlet->next = NULL;
			trackerlet->setBlockFeature(target);
			trackerlet->trackerletID = 0;//这个ID暂时没有什么意义，先临时放在这里

			//根据当前检测值对kalman进行修正
			measurement.at<float>(0) = (float)letTopLeftX;
			measurement.at<float>(1) = (float)letTopLeftY;
			measurement.at<float>(2) = (float)letWidth;
			measurement.at<float>(3) = (float)letHeight;
			KF.correct(measurement);//利用当前测量值对其滤波器进行修正

			if(trackerletHead != NULL)
			{
				delete trackerletHead;
			}
			trackerletHead = trackerlet;
			return false;
		}
	}
	else
	{
		//当前trackerletHead非空,可以尝试进行比较，另外如何进行预测呢？还没有明确的给出方案,这里将根据kalman滤波进行预测
		//后进行跟踪过程
		Mat prediction = KF.predict();//利用滤波器对当前检测tracklet矩形进行预测
		float *data = prediction.ptr<float>(0);
		int predictX = data[0];
		int predictY = data[1];
		int predictW = data[2];
		int predictH = data[3];
		std::cout<<predictX<<" "<<predictY<<" "<<predictW<<" "<<predictH<<std::endl;
		cv::Mat subImage = sourceImage(cv::Rect(predictX,predictY,predictW,predictH));
		cv::rectangle(sourceImage,cv::Rect(predictX,predictY,predictW,predictH),cv::Scalar(255,0,0),2);

		return true;
	}
}