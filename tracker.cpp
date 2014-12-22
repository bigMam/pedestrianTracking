#include "tracker.h"
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
	lockedPedArea = NULL;//不含头节点
	targetTrackerlet = NULL;//当前指向空
	//权重初始化操作
	for(int i = 0; i < 8; ++i)
	{
		weights[i] = 1.0 / 8;
	}
	front = 0;//初始化列表内容为空
	rear = 0;
	for(int i = 0; i < capacity; i++)
	{
		distratorList[i] = NULL;
	}
	letNumber = 0;//对得到trackerlet进行编号，编号可自加自减

	discriminator = Discriminator();//调用默认构造函数
}

//对残留内容进行删除操作，主要是distrator列表内容
Tracker::~Tracker()
{
	int traversal = rear;
	//清空列表
	while(traversal != front)
	{
		delete distratorList[traversal];
		traversal = (traversal + 1) % capacity;
	}
	delete targetTrackerlet;
}


void Tracker::setLoackedPedArea(LockedArea* result)
{
	lockedPedArea = result;//直接指向同一内存空间就好了，暂时先这样处理，后续可能会要进行改变
}

bool Tracker::update(cv::Mat &sourceImage)
{
	
	if(targetTrackerlet == 0)//当前targetTrackerlet为空,在最初跟踪目标确定阶段调用
	{
		if(lockedPedArea->next != NULL)//这里需要假定在视频的初始阶段有且仅有目标行人出现并可检测，这一限定条件
		{
			Trackerlet* trackerlet = new Trackerlet();
			extractTracklet(sourceImage,lockedPedArea->next,trackerlet);

			trackerlet->occupied++;
			targetTrackerlet = trackerlet;//这里因为之前没有存在traget，这里可以直接赋值，没有问题

			//在原图上进行绘制，红色表示测量值
			circle(sourceImage,cv::Point(trackerlet->topLeftX,trackerlet->topLeftY),5,CV_RGB(255,0,0),3);
			cv::rectangle(sourceImage,
					cv::Rect(trackerlet->topLeftX,trackerlet->topLeftY,targetTrackerlet->width,targetTrackerlet->height),
					cv::Scalar(255,0,0),2);
			//根据当前检测值对kalman进行修正，这里的predict必须添加，但这里的预测结果是没有不关心的，因为存在检测值
			Mat prediction = KF.predict();
			measurement.at<float>(0) = (float)trackerlet->topLeftX;
			measurement.at<float>(1) = (float)trackerlet->topLeftY;
			KF.correct(measurement);//利用当前测量值对其滤波器进行修正
			//这里直接认定检测得到就是正确的显然是存在问题的，只能这样认为，否则无法对跟踪目标进行指定

			//遍历删除操作
			LockedArea *head = lockedPedArea;
			LockedArea *current = lockedPedArea->next;
			while(current != NULL) 
			{
				LockedArea* tmp = current;
				head->next = current->next;
				delete tmp;
				current = head->next;
			}
			return false;
		}
		else
		{
			return true;//表示当前不存在检测矩形框，需要进一步的检测过程，
		}
	}
	else//当前targetTrackerlet存在，可以进行比较、预测过程
	{
		Trackerlet* newTargetTrackerlet = NULL;//用于存储新得到tracklet，无论是检测得到还是预测得到

		//首先对lockedPedArea进行遍历判断，两个结果：一个是发现targetTrackerlet，另外就是没有发现，好像废话，，，
		LockedArea* current = lockedPedArea->next;//遍历判断

		if(current != NULL)//当检测到行人时
		{
			while(current != NULL)
			{
				Trackerlet* trackerlet = new Trackerlet();
				extractTracklet(sourceImage,current,trackerlet);
				//认定为目标行人，阈值后续观察后再进行调节，这里需要加入位置信息

				if(isTargetTrackerlet(trackerlet))
				{
					//绘制
					circle(sourceImage,cv::Point(trackerlet->topLeftX,trackerlet->topLeftY),5,CV_RGB(255,0,0),3);

					//只能使用测量值对滤波器进行修正，不能够使用预测值进行修正，那样是不对的
					measurement.at<float>(0) = (float)trackerlet->topLeftX;
					measurement.at<float>(1) = (float)trackerlet->topLeftY;
					KF.correct(measurement);

					//临时存储
					newTargetTrackerlet = trackerlet;//仅仅是将当前相似trackerlet保存下来，还没有进行替换targetTrackerlet过程
					//move2next
					current = current->next;
					break;
				}
				else//加入distrator列表,同时将其加入到targetTarcker中后方
				{
					insertList(trackerlet);
					insertDistrator(trackerlet);
					//发现问题了，这里将内容存储到distrator中，同时存放到trakerletList，当一遍消除是，另一边将发生意外，
					//因此这里需要对双方进行约束，是进行删除，还是进行简单移除，需要进行判定，
					current = current->next;
				}
			}
			//将剩余tracklet放入distrator
			while(current != NULL)
			{
				Trackerlet* trackerlet = new Trackerlet();
				extractTracklet(sourceImage,current,trackerlet);

				insertList(trackerlet);
				insertDistrator(trackerlet);
				current = current->next;
			}
			//使用结束之后，清空lockedPedArea操作,遍历删除
			LockedArea *head = lockedPedArea;
			current = lockedPedArea->next;
			while(current != NULL)
			{
				LockedArea* tmp = current;
				head->next = current->next;
				delete tmp;
				current = head->next;
			}
		}
		
		if(newTargetTrackerlet == NULL)//之前经由lockedPedArea，并没有得到可用的trackerlet，需要经由kalman进行滤波预测
		{
			Mat prediction = KF.predict();//利用滤波器对当前检测tracklet矩形进行预测
			float *data = prediction.ptr<float>(0);
			int predictX = data[0];
			int predictY = data[1];
			cv::Mat subImage = sourceImage(cv::Rect(predictX,predictY,targetTrackerlet->width,targetTrackerlet->height));
			blockFeature target;
			extractor.computeFeature(subImage,target);
			//将当前得到blockfeature与之前存储内容进行比较
			double distinguishValue = this->distinguish(targetTrackerlet->featureSet,target);
			std::cout<<"预测trackerlet差异值为："<<distinguishValue<<std::endl;
			if(distinguishValue > 1.0)//当前预测结果并不能满足相似度要求，发出检测请求
				return true;
			else
			{
				//将预测结果保存下来
				Trackerlet* trackerlet = new Trackerlet();
				letNumber++;
				trackerlet->trackerletID = letNumber;//这个ID暂时没有什么意义，先临时放在这里
				trackerlet->topLeftX = predictX;
				trackerlet->topLeftY = predictY;
				trackerlet->width = targetTrackerlet->width;
				trackerlet->height = targetTrackerlet->height;
				trackerlet->setBlockFeature(target);
				trackerlet->next = NULL;
				trackerlet->occupied = 0;

				newTargetTrackerlet = trackerlet;//临时存储新近得到targetTrackerlet
				//在原图上进行绘制，绿色表示预测值
				circle(sourceImage,cv::Point(predictX,predictY),5,CV_RGB(0,255,0),3);
				cv::rectangle(sourceImage,
					cv::Rect(predictX,predictY,targetTrackerlet->width,targetTrackerlet->height),
					cv::Scalar(255,0,0),2);

				//这里需要仔细看一下，这样做可以么？会不会指向同一位置的内容被改变呢，不会的
			}
		}
		//这里将targetTrackerlet内容使用newTargetTrackerlet进行替代，同时保证原始列表内容不变
		newTargetTrackerlet->next = targetTrackerlet->next;

		targetTrackerlet->next = NULL;
		Trackerlet* tmp = targetTrackerlet;

		tmp->occupied--;
		if(tmp->occupied == 0)
		{
			delete tmp;//这里可以直接进行删除操作么？有没有可能被其他占有呢？可以直接删除
			letNumber--;
		}

		newTargetTrackerlet->occupied++;
		targetTrackerlet = newTargetTrackerlet;

		//终于到这里了，根据三方内容进行权重调节
		featureWeighting(targetTrackerlet->featureSet);

		//必定不会严格按照你的所有想法展开，因此在设计算法的时候要有足够的鲁棒性才可以

		//每个阶段对targetTrackerlet进行一次更新，这里的内容还不是特别明确，是否在每个阶段的末尾对list进行清空呢？
		//需要进行清空操作么？需要，那么对targettrackerlet需要怎样进行处理呢？保留，在之后还需要进行下一阶段的相似度判断
		return false;//表示不需要进行检测，可以继续进行下一次循环
	}
}

//deprecated
bool Tracker::update(cv::Mat &sourceImage,bool haveRectBoxing)
{
	//表示当前有新鲜出炉的行人检测矩形框，不需要进行预测过程？有待商榷
	if(haveRectBoxing && lockedPedArea != NULL)
	{

		Trackerlet* trackerlet = new Trackerlet();
		extractTracklet(sourceImage,lockedPedArea,trackerlet);
		circle(sourceImage,cv::Point(trackerlet->topLeftX,trackerlet->topLeftY),5,CV_RGB(255,0,0),3);

		//根据当前检测值对kalman进行修正，这里的predict必须添加，但这里的预测结果是没有不关心的，因为存在检测值
		Mat prediction = KF.predict();

		measurement.at<float>(0) = (float)trackerlet->topLeftX;
		measurement.at<float>(1) = (float)trackerlet->topLeftY;

		KF.correct(measurement);//利用当前测量值对其滤波器进行修正

		targetTrackerlet = trackerlet;//这里直接替换也是不正确的，替换需要在权重计算结束之后才进行
		//这里直接认定检测得到就是正确的显然是存在问题的，
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
		std::cout<<predictX<<" "<<predictY<<" "<<std::endl;

		if (targetTrackerlet == NULL)
			return true;
		else
		{
			//这里将根据当前得到tracklet与之前tracklet进行预测匹配，如果相似度可以则保留，否则删除，并发出检测请求
			//代码虽多但要思路清晰
			int letWidth = targetTrackerlet->width;
			int letHeight = targetTrackerlet->height;
			cv::Mat subImage = sourceImage(cv::Rect(predictX,predictY,letWidth,letHeight));
			blockFeature target;
			extractor.computeFeature(subImage,target);

			//将当前得到blockfeature与之前存储内容进行比较
			double distinguish = this->distinguish(targetTrackerlet->featureSet,target);
			std::cout<<"差异值为："<<distinguish<<std::endl;
			if(distinguish > 0.35)
				return true;
			else
			{
				circle(sourceImage,cv::Point(predictX,predictY),5,CV_RGB(0,255,0),3);
				cv::rectangle(sourceImage,cv::Rect(predictX,predictY,letWidth,letHeight),cv::Scalar(255,0,0),2);
				return false;
			}
		}
	}
}

//这里存在一个反复出现的内容，可以提取作为一个单独函数实现
//根据矩形框，提取tacklet
void Tracker::extractTracklet(cv::Mat &sourceImage,LockedArea* lockedPedArea,Trackerlet* trackerlet)
{
	int topLeftX = lockedPedArea->topLeftX;
    int topLeftY = lockedPedArea->topLeftY;
	int width = lockedPedArea->width;
	int height = lockedPedArea->height;
	cv::Rect rect(topLeftX,topLeftY,width,height);

	int letWidth = rect.width * 0.4;
	int letHeight = rect.height * 0.18;
	int letTopLeftX = rect.x + rect.width * 0.3;
	int letTopLeftY = rect.y + rect.height * 0.25;
	cv::Mat subImage = sourceImage(cv::Rect(letTopLeftX,letTopLeftY,letWidth,letHeight));

	blockFeature target;
	extractor.computeFeature(subImage,target);

	letNumber++;
	trackerlet->trackerletID = letNumber;//这个ID暂时没有什么意义，先临时放在这里
	trackerlet->topLeftX = letTopLeftX;
	trackerlet->topLeftY = letTopLeftY;
	trackerlet->width = letWidth;
	trackerlet->height = letHeight;
	trackerlet->setBlockFeature(target);
	trackerlet->next = NULL;

	trackerlet->occupied = 0;

	//对矩形框进行标定
	cv::rectangle(sourceImage,cv::Rect(letTopLeftX,letTopLeftY,letWidth,letHeight),cv::Scalar(255,0,0),2);
}

bool Tracker::isTargetTrackerlet(Trackerlet* current)
{
	if(targetTrackerlet == NULL)
		return false;
	//根据位置信息及feature进行判断过程
	//加入位置信息，位置相近可以讲阈值放宽，位置差别大的量目标，则需要有较高的相似度，才认定为同一目标
	//位置相近时设定为1.0，位置差别较大时设定为0.5，可以考虑一下
	double distinguishValue = this->distinguish(targetTrackerlet->featureSet,current->featureSet);
	std::cout<<"差异值为："<<distinguishValue<<std::endl;
	if(distinguishValue < 0.5)
		return true;
	else
	{
		//这里进行位置判断的依据是什么，怎样才认定为两目标接近？
		//最原始的方法位置差值小于给定值
		int diffX = std::abs(targetTrackerlet->topLeftX  - current->topLeftX);
		int diffY = std::abs(targetTrackerlet->topLeftY - current->topLeftY);
		if(diffX < 30 && diffY < 30)//认为两者较为接近
			if(distinguishValue < 1.0)
				return true;
			else
				return false;
		else
			if(distinguishValue < 0.5)
				return true;
			else
				return false;
	}
}

//将tracket插入到distrator列表中，并保证容量不超过上限，在超出时，能够删除last one
void Tracker::insertDistrator(Trackerlet* trackerlet)
{
	//可以使用队列的形式，FIFO，符合队列操作一般特性，
	//操作到后面，往往是利用新添加元素将指定元素进行替换，在队列已满的情形下，也是能够快速插入的


	//注：front 指向队头后一个元素，rear指向队尾元素
	trackerlet->occupied++;
	if((front + 1)%capacity == rear)//队满
	{	
		//插入之前需要将队尾元素出队	
		Trackerlet* tmp = distratorList[rear];
		
		tmp->occupied--;
		if(tmp->occupied == 0)
		{
			delete tmp;//有必要则将元素删除
			letNumber--;
		}
		rear = (rear + 1)%capacity;

		distratorList[front] = trackerlet;
		front = (front + 1)%capacity;
		
		//这里就比链表要方便一些，因为仅仅是存储了指针信息，可以方便的进行赋值操作
	}
	else//队未满可以直接进行插入操作
	{
		distratorList[front] = trackerlet;
		front = (front + 1)%capacity;
	}
}

//计算当前特征与目标特征之间的差值
double Tracker::distinguish(blockFeature& target, blockFeature& current)
{
	discriminator.clearDistance();
	discriminator.setCurrentFeature(current);
	discriminator.computeDistance(target);
	double dissimilarity = discriminator.distinguish();

	return dissimilarity;
}
//根据当前current（正确目标），preTarget（先前存储的），distrator，已抛弃内容，
void Tracker::featureWeighting(blockFeature& current)
{
	//根据论文中给出的公式分别计算各个巴氏距离
	discriminator.clearDistance();
	discriminator.setCurrentFeature(current);

	double distance[8];
	double meanDistance[8];
	for(int i = 0; i < 8; i++)
	{
		distance[i] = 0;
		meanDistance[i] = 0;
	}
	int traversal = rear;
	while(traversal != front)
	{   
		blockFeature distratorFeature = distratorList[traversal]->featureSet;
		discriminator.computeDistanceHold(distratorFeature);
		traversal = (traversal + 1) % capacity;
	}
	if(rear != front)//表示当前distrator列表非空
	{
		discriminator.getDistance(meanDistance);
		int count = rear < front ? front - rear : front - rear + capacity;
		for(int i = 0; i < 8; i++)
		{
			meanDistance[i] = meanDistance[i] / count;//获得与弃用者平均巴氏距离
		}
	}
	blockFeature targetFeature = targetTrackerlet->featureSet;
	discriminator.computeDistance(targetFeature);
	discriminator.getDistance(distance);//获取与跟踪目标巴氏距离
	if(meanDistance[0] != 0)
	{
		for(int i =0; i < 8; i++)
		{
			weights[i] = distance[i] != 0 ? weights[i] + (meanDistance[i] -  distance[i]) : weights[i] + meanDistance[i];
		}
	}
	double sum = 0;
	for(int i = 0; i < 8; ++i)
	{
		sum = sum + weights[i];
	}
	for(int i = 0; i < 8; ++i)
	{ 
		weights[i] = weights[i] / sum;
	}
	discriminator.setWeights(weights);
}

Trackerlet* Tracker::getTrackerlist()
{
	return targetTrackerlet;
}


//利用反馈结果对targetTrackerlet进行修正
void Tracker::correctTarget(Trackerlet* correctTrackerlet)
{
	targetTrackerlet = correctTrackerlet;//这样直接进行修正是存在风险的，一是之前内容没有delete，
	//二是所指向的内容很有可能会在之后的某个时间点被清除，从而存在targetTrackerlet指向NULL的情况发生
}

//将元素插入链表，这里如果使用STL会更加方便一些？先实现，这个属于后期的优化过程？是这样么
void Tracker::insertList(Trackerlet* trackerlet)
{
	trackerlet->occupied++;
	//简单的头插法实现插入操作
	trackerlet->next = targetTrackerlet->next;
	targetTrackerlet->next = trackerlet;
}

void Tracker::clearList()//对targetTrackerlet列表进行清空操作，仅保留首个节点
{
	if(targetTrackerlet == NULL)
		return;
	Trackerlet *curr,*tmp;
	curr = targetTrackerlet->next;
	while(curr != NULL)//对列表进行清空操作
	{
		tmp = curr;
		targetTrackerlet->next = curr->next;
		tmp->occupied--;
		if(tmp->occupied == 0)
		{
			delete tmp;
			tmp = NULL;
			letNumber--;
		}
		curr = targetTrackerlet->next;
	}
}