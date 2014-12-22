#include "manager.h"

Manager::Manager()
{
	front = 0;//初始化列表内容为空
	rear = 0;
	for(int i = 0; i < capacity; i++)
	{
		targetPool[i] = NULL;
	}
}

void Manager::setTrackerletList(Trackerlet* list)
{
	trackerletList = list;
}
//根据传递trackerlet判定当前跟踪目标是否为指定目标，同时对自身信息进行更新
//在跟踪目标与指定目标不一致的情形下（return false），对tracker进行修正
//默认链表第一个节点为跟踪目标
bool Manager::dicision()
{

	return true;
}