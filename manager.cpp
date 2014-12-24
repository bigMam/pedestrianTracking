#include "manager.h"


Manager::Manager()
{
	front = 0;//初始化列表内容为空
	rear = 0;
	for(int i = 0; i < capacity; i++)
	{
		targetPool[i] = NULL;
	}
	discriminator = Discriminator();//调用默认构造函数
	acc = 0;
}

void Manager::setTrackerletList(Trackerlet* list)
{
	trackerletList = list;
}

void Manager::updateWeights(double weights[])
{
	discriminator.setWeights(weights);
}

//根据传递trackerlet判定当前跟踪目标是否为指定目标，同时对自身信息进行更新
//在跟踪目标与指定目标不一致的情形下（return false），对tracker进行修正
//默认链表第一个节点为跟踪目标
//返回值表示，头结点是否为跟踪节点。是 true，否 false
bool Manager::dicision()
{
	//完成两个操作，判定头结点是否为指定目标；更新目标池

	//判断当前trackerletList头结点是否为跟踪节点
	if(trackerletList == NULL)
		return true;//在链表无内容情形下，自然无法进行修正过程，自然返回true，都不知道有没有错，怎么改？有道理哈
	
	if(front == rear)//当前目标池为空，直接加入链表头结点
	{
		trackerletList->occupied++;
		targetPool[front++] = trackerletList;//将头结点内容入池
		return true;
	}
	else//目标池非空的情形下，需要对头结点进行判定，
	{
		discriminator.clearDistance();
		//首先将头结点blockfeature内容暂时存放于discriminator
		discriminator.setCurrentFeature(trackerletList->featureSet);

		//将头结点内容与目标池中存储内容进行比较
		int traversal = rear;
		while(traversal != front)
		{   
			blockFeature distratorFeature = targetPool[traversal]->featureSet;
			discriminator.computeDistanceHold(distratorFeature);
			traversal = (traversal + 1) % capacity;
		}
		double dissimilarity = discriminator.distinguish();//得到当前target与目标池中所有元素的差异值
		if(dissimilarity > 1.0)//表示头结点不是跟踪目标,需要对list剩余内容进行遍历，寻找是否存在跟踪目标
		{
			correctTarget = haveTarget();//得到修正trackerlet
			if(correctTarget != NULL)
			{
				updatePool(correctTarget);
			}
			return false;
		}
		else
		{
			//认定当前头结点内容为指定跟踪目标，在必要的情形下对目标池进行更新
			updatePool(trackerletList);
			return true;
		}
	}
}

//判定trackerlet链表剩余内容中是否存在之前指定跟踪目标，有则返回指向指针，否则返回NULL
Trackerlet* Manager::haveTarget()
{
	Trackerlet* current = trackerletList->next;
	while(current != NULL)
	{
		discriminator.clearDistance();
		//首先将头结点blockfeature内容暂时存放于discriminator
		discriminator.setCurrentFeature(trackerletList->featureSet);

		//将头结点内容与目标池中存储内容进行比较
		int traversal = rear;
		while(traversal != front)
		{   
			blockFeature distratorFeature = targetPool[traversal]->featureSet;
			discriminator.computeDistanceHold(distratorFeature);
			traversal = (traversal + 1) % capacity;
		}
		double dissimilarity = discriminator.distinguish();//得到当前target与目标池中所有元素的差异值
		if(dissimilarity < 1.0)
		{
			return current;
		}
		else
		{
			current = current->next;
		}
	}
	return NULL;
}

//到现在为止，思考内容还是面向过程的思想，，哎
//对目标池进行更新，明确target的前提下，类似之前的insertDistrator函数，需要对时间间隔进行掌控
//输入参数为当前确定target
/***********可改进，标记一下**********/
void Manager::updatePool(Trackerlet* target)
{
	//这里需要设定更新间隔
	//注：front 指向队头后一个元素，rear指向队尾元素
	acc++;
	if(acc > interval)
	{
		acc = 0;//对累加器进行清零操作，这里还是用文章可以做的，是仅仅将oldest删除，还是设定一些判定依据，
		//为了保证目标池内容的多样性，将最为接近的一个进行删除操作，这样的话，使用链表会比较方便，毕竟涉及到了
		//任意删除操作，之后再进行考虑吧，mark...
		target->occupied++;
		if((front + 1)%capacity == rear)//队满
		{	
			//插入之前需要将队尾元素出队	
			Trackerlet* tmp = targetPool[rear];
			tmp->occupied--;
			if(tmp->occupied == 0)
			{
				delete tmp;//有必要则将元素删除
				tmp = NULL;
			}
			rear = (rear + 1)%capacity;

			targetPool[front] = target;
			front = (front + 1)%capacity;
		
			//这里就比链表要方便一些，因为仅仅是存储了指针信息，可以方便的进行赋值操作
		}
		else//队未满可以直接进行插入操作
		{
			targetPool[front] = target;
			front = (front + 1)%capacity;
		}
	}
}

Trackerlet* Manager::correct()
{
	return correctTarget;
}