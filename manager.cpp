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
	//完成两个操作，判定头结点是否为指定目标；更新目标池，当前设定目标池容量为5，可以增加，

	//判断当前trackerletList头结点是否为跟踪节点
	if(trackerletList == NULL)
		return true;//在链表无内容情形下，自然无法进行修正过程，自然返回true，都不知道有没有错，怎么改？
	if(front == rear)//当前目标池为空，直接加入链表头结点
	{
		trackerletList->occupied++;
		targetPool[front++] = trackerletList;//将头结点内容入池
	}
	else//目标池非空的情形下，需要对头结点进行判定，
	{
		

	}




	//对目标池进行更新操作



	return true;
}