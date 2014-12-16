#include "myTracker.h"
//将检测得到pedArea存储到tracker中，这里不能直接使用引用吧，
//假设是使用同一内存空间，则后续的改变，将对其造成影响。，貌似没有问题，等出现问题再进行修改吧

#define HAVE_BORDER 1
//使用边界预测，但是对其不进行赋值，对角坐标预测效果反而不错，暂时不追究其原因，先将整个代码跑通，框架搭好先

Tracker::Tracker()
{
	//完成对kalman滤波器的初始化操作，用于之后的tracklet的预测过程

#if HAVE_BORDER == 1
	stateNum = 8;
	measureNum = 4;
#else
	stateNum = 4;
	measureNum = 2;
#endif

	KF = cv::KalmanFilter(stateNum, measureNum, 0);
	state = cv::Mat(stateNum, 1, CV_32F);//滤波器状态矩阵
	processNoise = cv::Mat(stateNum, 1, CV_32F);//滤波器处理噪声
	measurement = cv::Mat::zeros(measureNum, 1, CV_32F);//滤波器测量矩阵

#if HAVE_BORDER == 1
	KF.transitionMatrix = *( Mat_<float>(8, 8) << 
		1,0,1,0,0,0,0,0,
		0,1,0,1,0,0,0,0,
		0,0,1,0,0,0,0,0,
		0,0,0,1,0,0,0,0,
		0,0,0,0,1,0,1,0,
		0,0,0,0,0,1,0,1,
		0,0,0,0,0,0,1,0,
		0,0,0,0,0,0,0,1);//转移矩阵
#else
	KF.transitionMatrix = *(Mat_<float>(4,4) << 1,0,1,0,0,1,0,1,0,0,1,0,0,0,0,1);
#endif

	setIdentity(KF.measurementMatrix);//测量矩阵
	setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));
	setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, cv::Scalar::all(1));
	//KF.statePost = *(Mat_<float>(4,1) << 320,240,1,1);
	randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));

	//对二次特征提取模块进行初始化操作
	extractor = FeatureExtractor();
	extractor.initCache();
	lockedPedArea = NULL;
	distrator = NULL;
	targetTrackerlet = Trackerlet();

	//权重初始化操作
	for(int i = 0; i < 8; ++i)
	{
		weights[i] = 1.0 / 8;
	}

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
//应该而且很有必要将边框信息考虑进来？直接使用之前的边框信息不可以？
//关于左上角点的预测还是有一定效果的
bool Tracker::update(cv::Mat &sourceImage,bool haveRectBoxing)
{
	//表示当前有新鲜出炉的行人检测矩形框，不需要进行预测过程？有待商榷
	if(haveRectBoxing && lockedPedArea != NULL)
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

		Trackerlet trackerlet = Trackerlet();
		trackerlet.topLeftX = letTopLeftX;
		trackerlet.topLeftY = letTopLeftY;
		trackerlet.width = letWidth;
		trackerlet.height = letHeight;
		trackerlet.next = NULL;
		trackerlet.setBlockFeature(target);
		trackerlet.trackerletID = 0;//这个ID暂时没有什么意义，先临时放在这里
		circle(sourceImage,cv::Point(letTopLeftX,letTopLeftY),5,CV_RGB(255,0,0),3);//将当前测量值直接在原图上进行绘制

		//根据当前检测值对kalman进行修正，这里的predict必须添加，但这里的预测结果是没有不关心的，因为存在检测值
		Mat prediction = KF.predict();

		measurement.at<float>(0) = (float)letTopLeftX;
		measurement.at<float>(1) = (float)letTopLeftY;
#if HAVE_BORDER == 1
		//measurement.at<float>(2) = (float)letWidth;
		//measurement.at<float>(3) = (float)letHeight;//存在疑问，这里为什么不能加上这里的测量值，理论上对x、y值是没有影响的
		//需要通过阅读源码对其进行解释？？？？存疑，但是还是需要继续走下去
#endif
		KF.correct(measurement);//利用当前测量值对其滤波器进行修正

		targetTrackerlet = trackerlet;//这里直接替换也是不正确的，替换需要在权重计算结束之后才进行
		return false;

	}
	else
	{
		//当前trackerletHead非空,可以尝试进行比较，另外如何进行预测呢？
		//后进行跟踪过程
		Mat prediction = KF.predict();//利用滤波器对当前检测tracklet矩形进行预测
		float *data = prediction.ptr<float>(0);
		int predictX = data[0];
		int predictY = data[1];
#if HAVE_BORDER == 1
		int predictW = data[2];//这里的边框值暂时不需要，而是选择使用tracklet的边框进行取值
		int predictH = data[3];
#endif
		std::cout<<predictX<<" "<<predictY<<" "<<std::endl;

		if (targetTrackerlet.topLeftX == 0)
			return true;
		else
		{
			//这里将根据当前得到tracklet与之前tracklet进行预测匹配，如果相似度可以则保留，否则删除，并发出检测请求
			//代码虽多但要思路清晰
			int letWidth = targetTrackerlet.width;
			int letHeight = targetTrackerlet.height;
			cv::Mat subImage = sourceImage(cv::Rect(predictX,predictY,letWidth,letHeight));
			blockFeature target;
			extractor.computeFeature(subImage,target);

			//将当前得到blockfeature与之前存储内容进行比较
			double distinguish = this->distinguish(targetTrackerlet.featureSet,target);
			std::cout<<"差异值为："<<distinguish<<std::endl;
			if(distinguish > 0.35)
				return true;
			else
			{
				//对tracklet进行更新过程，含权重调整，这里的
				//trackerletHead->setBlockFeature(target);
				//trackerletHead->topLeftX = predictX;
				//trackerletHead->topLeftY = predictY;
				//measurement.at<float>(0) = (float)predictX;
				//measurement.at<float>(1) = (float)predictY;
				//KF.correct(measurement);这里直接用预测值来进行修正显然是不合理的
				circle(sourceImage,cv::Point(predictX,predictY),5,CV_RGB(0,255,0),3);
				cv::rectangle(sourceImage,cv::Rect(predictX,predictY,letWidth,letHeight),cv::Scalar(255,0,0),2);
				return false;
			}
		}
	}
}
//计算当前特征与目标特征之间的差值
double Tracker::distinguish(blockFeature& target, blockFeature& current)
{
	cv::MatND targetLBP = cv::Mat(target.cs_lbpFeature);
	cv::MatND currentLBP = cv::Mat(current.cs_lbpFeature);
	cv::MatND targetCanny = cv::Mat(target.cannyFeature);
	cv::MatND currentCanny = cv::Mat(current.cannyFeature);

	double hueDistance = compareHist(target.hueHist,current.hueHist,CV_COMP_BHATTACHARYYA);
	double satDistance = compareHist(target.satHist,current.satHist,CV_COMP_BHATTACHARYYA);
	double valDistance = compareHist(target.valHist,current.valHist,CV_COMP_BHATTACHARYYA);
	double lbpDistance = compareHist(targetLBP,currentLBP,CV_COMP_BHATTACHARYYA);
	double cannyDistance = compareHist(targetCanny,currentCanny,CV_COMP_BHATTACHARYYA);
	double horDerDistance = compareHist(target.horDerHist,current.horDerHist,CV_COMP_BHATTACHARYYA);
	double verDerDistance = compareHist(target.verDerHist,current.verDerHist,CV_COMP_BHATTACHARYYA);

	cv::MatND targetEHD = cv::Mat(5,1,CV_32F);
	cv::MatND currentEHD = cv::Mat(5,1,CV_32F);
	for(int i = 0; i < 5; i++)
	{
		float* targetPtr = targetEHD.ptr<float>(i);
		float* currentPtr = currentEHD.ptr<float>(i);
		targetPtr[0] = target.EHD[i];
		currentPtr[0] = current.EHD[i];
	}
	double EHDDistance = compareHist(targetEHD,currentEHD,CV_COMP_BHATTACHARYYA);
	//完成距离计算过程，

	//计算当前图像块与目标图像块的差异值
	double dissimilarity = weights[0] * hueDistance + weights[0] * satDistance + weights[0] * valDistance + 
		weights[0] * lbpDistance + weights[0] * cannyDistance + weights[0] * horDerDistance + 
		weights[0] * verDerDistance + weights[0] * EHDDistance;

	std::cout<<"dissimilarity is :"<<dissimilarity<<std::endl;
	return dissimilarity;
}
//根据当前current（正确目标），preTarget（先前存储的），distrator，已抛弃内容，
void Tracker::featureWeighting(blockFeature& current)
{
	//根据论文中给出的公式分别计算各个巴氏距离
	cv::MatND targetEHD = cv::Mat(5,1,CV_32F);
	cv::MatND currentEHD = cv::Mat(5,1,CV_32F);

	//完成current距离与所有distrator的feature巴氏距离均值
	double meanhueDistance = 0,meansatDistance = 0,meanvalDistance = 0;
	double meanlbpDistance = 0,meancannyDistance = 0;
	double meanhorDerDistance = 0,meanverDerDistance = 0,meanEHDDistance = 0;

	cv::MatND currentLBP = cv::Mat(current.cs_lbpFeature);
	cv::MatND currentCanny = cv::Mat(current.cannyFeature);

	int count = 0;
	Trackerlet *distratorPtr = distrator;
	while(distratorPtr != NULL)
	{
		cv::MatND targetLBP = cv::Mat(distratorPtr->featureSet.cs_lbpFeature);
		cv::MatND targetCanny = cv::Mat(distratorPtr->featureSet.cannyFeature);

		meanhueDistance = meanhueDistance + compareHist(distratorPtr->featureSet.hueHist,current.hueHist,CV_COMP_BHATTACHARYYA);
		meansatDistance = meansatDistance + compareHist(distratorPtr->featureSet.satHist,current.satHist,CV_COMP_BHATTACHARYYA);
		meanvalDistance = meanvalDistance + compareHist(distratorPtr->featureSet.valHist,current.valHist,CV_COMP_BHATTACHARYYA);
		meanlbpDistance = meanlbpDistance + compareHist(targetLBP,currentLBP,CV_COMP_BHATTACHARYYA);
		meancannyDistance = meancannyDistance + compareHist(targetCanny,currentCanny,CV_COMP_BHATTACHARYYA);
		meanhorDerDistance = meanhorDerDistance + compareHist(distratorPtr->featureSet.horDerHist,current.horDerHist,CV_COMP_BHATTACHARYYA);
		meanverDerDistance = meanverDerDistance + compareHist(distratorPtr->featureSet.verDerHist,current.verDerHist,CV_COMP_BHATTACHARYYA);
		for(int i = 0; i < 5; i++)
		{
			float* targetPtr = targetEHD.ptr<float>(i);
			float* currentPtr = currentEHD.ptr<float>(i);
			targetPtr[0] = distratorPtr->featureSet.EHD[i];
			currentPtr[0] = current.EHD[i];
		}
		meanEHDDistance = meanEHDDistance + compareHist(targetEHD,currentEHD,CV_COMP_BHATTACHARYYA);
	}
	if(count != 0)
	{
		meanhueDistance = meanhueDistance / count;
		meansatDistance = meansatDistance / count;
		meanvalDistance = meanvalDistance / count;
		meanlbpDistance = meanlbpDistance / count;
		meancannyDistance = meancannyDistance / count;
		meanhorDerDistance = meanhorDerDistance / count;
		meanverDerDistance = meanverDerDistance / count;
		meanEHDDistance = meanEHDDistance / count;
	}
	
	//完成current与preTarget的feature巴氏距离的计算
	double hueDistance = 0,satDistance = 0,valDistance = 0;
	double lbpDistance = 0,cannyDistance = 0;
	double horDerDistance = 0,verDerDistance = 0,EHDDistance = 0;

	cv::MatND targetLBP = cv::Mat(targetTrackerlet.featureSet.cs_lbpFeature);
	cv::MatND targetCanny = cv::Mat(targetTrackerlet.featureSet.cannyFeature);

	hueDistance = compareHist(targetTrackerlet.featureSet.hueHist,current.hueHist,CV_COMP_BHATTACHARYYA);
	satDistance = compareHist(targetTrackerlet.featureSet.satHist,current.satHist,CV_COMP_BHATTACHARYYA);
	valDistance = compareHist(targetTrackerlet.featureSet.valHist,current.valHist,CV_COMP_BHATTACHARYYA);
	lbpDistance = compareHist(targetLBP,currentLBP,CV_COMP_BHATTACHARYYA);
	cannyDistance = compareHist(targetCanny,currentCanny,CV_COMP_BHATTACHARYYA);
	horDerDistance = compareHist(targetTrackerlet.featureSet.horDerHist,current.horDerHist,CV_COMP_BHATTACHARYYA);
	verDerDistance = compareHist(targetTrackerlet.featureSet.verDerHist,current.verDerHist,CV_COMP_BHATTACHARYYA);

	for(int i = 0; i < 5; i++)
	{
		float* targetPtr = targetEHD.ptr<float>(i);
		float* currentPtr = currentEHD.ptr<float>(i);
		targetPtr[0] = targetTrackerlet.featureSet.EHD[i];
		currentPtr[0] = current.EHD[i];
	}
	EHDDistance = compareHist(targetEHD,currentEHD,CV_COMP_BHATTACHARYYA);
	
	//完成对feature的歌权重调整过程，这里仅仅是一种方法，但这是否是最好的方法呢，还有待进一步的确定
	weights[0] = weights[0] + (meanhueDistance - hueDistance);
	weights[1] = weights[1] + (meansatDistance - satDistance);
	weights[2] = weights[2] + (meanvalDistance - valDistance);
	weights[3] = weights[3] + (meanlbpDistance - lbpDistance);
	weights[4] = weights[4] + (meancannyDistance - cannyDistance);
	weights[5] = weights[5] + (meanhorDerDistance - horDerDistance);
	weights[6] = weights[6] + (meanverDerDistance - verDerDistance);
	weights[7] = weights[7] + (meanEHDDistance - EHDDistance);

	//归一化操作
	double sum = 0;
	for(int i = 0; i < 8; ++i)
	{
		sum = sum + weights[i];
	}
	for(int i = 0; i < 8; ++i)
	{
		weights[i] = weights[i] / sum;
	}
	//完成权重调整，但是还不知道效果如何
}