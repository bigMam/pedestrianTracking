
//实现对trackerlet的管理工作，在必要的时刻完成跟踪目标的遮挡处理，
//能够在目标再次出现时，快速定位
#include "tracker.h"
class Manager
{
private:
	Trackerlet* trackerletList;//经由检测层传递过来的trackerlet列表
	Trackerlet targetTrackerlet[6];//对特定目标进行存储，用于对特定目标进行判定过程，同时在遮挡结束后，能够根据历史信息
	//将目标再次识别，仅仅依靠上一次信息是不能够将目标判定出来的（在遮挡结束后环境与上次是存在较多不同）
public:
	void setTrackerletList(Trackerlet* list);

	bool dicision(){return true;};//根据传递trackerlet判定当前跟踪目标是否为指定目标，同时对自身信息进行更新
	//在跟踪目标与指定目标不一致的情形下（return false），对tracker进行修正

	Trackerlet* correct(){return NULL;};//如果经由决策，认定当前tracker跟踪目标与指定目标不同，则进行修正，也就是将当前指定tracklet反馈给tracker

};