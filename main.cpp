#include <iostream>
#include "mySVM.h"
#include "SymmetryProcess.h"
#include "myTracker.h"
using namespace cv;
int main()
{
	const char* filename =  "D:\\ImageDataSets\\TestSamples\\image1202.jpg";
	const char* videoname = "D:\\ImageDataSets\\trackingSamples\\MVI_2693_01.avi";
	SVMDetector detector;
	detector.loadDetectorVector("mydetectorNew.xml");
	detector.initSymmetryParam(527,531,310,248,530,0.72);

	Tracker tracker = Tracker();

	VideoCapture cap(videoname);
	if(!cap.isOpened())
		return -1;
	cv::Mat sourceImage;
	cv::Mat gray;
	int interval = 8;//detector检测调用间隔
	int k = 0;//统计调用间隔
	bool isRequest = true;
	LockedArea* current,*tmp;//记录当前已经检测得到的行人

	while(cap.read(sourceImage))
	{
		//调用检测过程，需要进行 行人检测过程
		if(k > interval || isRequest)
		{
			cv::cvtColor(sourceImage,gray,CV_BGR2GRAY);
			cv::imshow("frame",sourceImage);
			detector.detectBaseOnSymmetry(gray);
			tmp = detector.getResultRect();
			current = tmp;
			while(tmp != NULL)
			{
				Rect rect = Rect(tmp->topLeftX,tmp->topLeftY,tmp->width,tmp->height);
				cv::rectangle(sourceImage,rect,Scalar(0,0,0),1);
				tmp= tmp->next;
			}
			tracker.setLoackedPedArea(current);//将当前得到的结果存储到tracker中，用于生成新的tracklet
			k = 0;
			isRequest = tracker.update(sourceImage,true);
		}
		else
		{
			isRequest = tracker.update(sourceImage,false);
		}

		imshow("sourceImage",sourceImage);
		if(!isRequest)//当前tracklet更新成功，可以进行tracklet管理过程
		{
			//将更新得到tracklet用于后续的manager，这个后续再继续进行吧
			//git测试代码
			//std::cout<<"github test"<<std::endl;
		}
		if(cv::waitKey(1) == 27)
			break;
		k++;
	}
	cv::waitKey(0);
	return 0;
}