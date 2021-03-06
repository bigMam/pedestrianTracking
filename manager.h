
//实现对trackerlet的管理工作，在必要的时刻完成跟踪目标的遮挡处理，
//能够在目标再次出现时，快速定位
#pragma once
#include "tracker.h"
#include "discriminator.h"

//问题是这里与tracker使用的是同一个分辨器，有必要分成两部分么？
//显然是不能分开的，但是不分开能够很好的完成么？不能，这样可以将weights进行刷新到当前discriminator中
class Manager
{
private:
	Trackerlet* trackerletList;//经由检测层传递过来的trackerlet列表

	Trackerlet* targetPool[6];//目标池，存放已识别目标的历史信息，多个可能，间隔时间x将目标入池，同时剔除oldest
	Trackerlet* correctTarget;
	static const int capacity = 6;//目标池容量信息
	int front;
	int rear;
	static const int interval = 30;//静态常量，目标入池的时间间隔
	int acc;//用于对请求更新次数进行累加，
	Discriminator discriminator;//分辨器，对两trackerlet进行差异值计算

	//对特定目标进行存储，用于对特定目标进行判定过程，同时在遮挡结束后，能够根据历史信息
	//将目标再次识别，仅仅依靠上一次信息是不能够将目标判定出来的（在遮挡结束后环境与上次是存在较多不同）
public:
	Manager();

	void setTrackerletList(Trackerlet* list);
	void updateWeights(double weights[]);//对分辨器特征权重进行更新
	bool dicision();//根据传递trackerlet判定当前跟踪目标是否为指定目标，同时对自身信息进行更新
	//在跟踪目标与指定目标不一致的情形下（return false），对tracker进行修正
	Trackerlet* haveTarget();//遍历list寻找是否存在跟踪目标

	Trackerlet* correct();//如果经由决策，认定当前tracker跟踪目标与指定目标不同，则进行修正，也就是将当前指定tracklet反馈给tracker

	void updatePool(Trackerlet* target);//对目标池进行更新
};