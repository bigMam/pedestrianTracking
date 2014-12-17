#include <iostream>
#include "mySVM.h"
#include "SymmetryProcess.h"
#include "myTracker.h"
using namespace cv;

extern int videoCut(const char* sourceVideo,const char* targetVideo,int stratSec,int endSec);
int main()
{
	const char* filename =  "D:\\ImageDataSets\\TestSamples\\image1202.jpg";
	const char* videoname = "D:\\ImageDataSets\\trackingSamples\\MVI_2708_75_2.avi";

	cv::VideoCapture cap(videoname);
	if(!cap.isOpened())
		return -1;
	SVMDetector detector;
	detector.loadDetectorVector("mydetectorNew.xml");
	detector.initSymmetryParam(527,531,310,248,530,0.75);

	Tracker tracker = Tracker();//distinguish是在tracker中完成的，


	cv::Mat sourceImage;
	cv::Mat gray;
	int interval = 10;//detector检测调用间隔
	int k = 0;//统计调用间隔
	bool isRequest = true;//检测调用请求
	LockedArea* current,*tmp;//记录当前已经检测得到的行人

	while(cap.read(sourceImage))
	{
		//检测调用过程，完成行人检测过程
		if( k > interval || isRequest )//两种情况调用：周期调用，响应请求
		{
			cv::cvtColor(sourceImage,gray,CV_BGR2GRAY);
			//cv::imshow("frame",sourceImage);
			detector.detectBaseOnSymmetry(gray);
			tmp = detector.getResultRect();//获得行人检测结果，
			current = tmp;
			//对检测内容进行绘制，前提是能够检测得到行人
			while(tmp != NULL)
			{
				Rect rect = Rect(tmp->topLeftX,tmp->topLeftY,tmp->width,tmp->height);
				cv::rectangle(sourceImage,rect,Scalar(0,0,0),1);
				tmp= tmp->next;
			}
			//这里直接将指针进行传递，这样是否可行？这样的话，tracker中始终是最新的检测结果，在两次检测期间
			//则保存上次检测结果，这样有什么用呢
			tracker.setLoackedPedArea(current);//将当前得到的结果存储到tracker中，用于生成新的tracklet
			k = 0;
			isRequest = tracker.update(sourceImage,true);//进行更新，根据更新结果判断是否需要进一步的进行检测
		}
		else
		{
			isRequest = tracker.update(sourceImage,false);
		}

		imshow("sourceImage",sourceImage);
		if(!isRequest)//当前tracklet更新成功，可以进行tracklet管理过程
			//如果更新成功则进行传递tracklet，
		{
			//将更新得到tracklet用于后续的manager，这个后续再继续进行吧
			//git测试代码
			//std::cout<<"github test"<<std::endl;
		}

		char key = cv::waitKey(3);
		if(key == 27)
			break;
		if(key == 32)
		{
			while(cv::waitKey(3) != 32);
		}
		
		k++;
	}
	cv::waitKey(0);
	return 0;
}