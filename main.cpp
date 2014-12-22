#include <iostream>
#include "mySVM.h"
#include "SymmetryProcess.h"
#include "tracker.h"
#include "manager.h"
using namespace cv;

extern int videoCut(const char* sourceVideo,const char* targetVideo,int stratSec,int endSec);
int pedTracking(const char* videoname);
int main()
{
	const char* videoname = "D:\\ImageDataSets\\trackingSamples\\MVI_2708_75_2.avi";
	//const char* targetvideo = "D:\\ImageDataSets\\trackingSamples\\MVI_2722_target_2.avi";

	//videoCut(videoname,targetvideo,1,20);

	//const char* targetVideo = "D:\\ImageDataSets\\trackingSamples\\MVI_2708_75_2_target.avi";
	//int ex=static_cast<int>(cap.get(CV_CAP_PROP_FOURCC)); 
	//char EXT[] = {ex & 0XFF , (ex & 0XFF00) >> 8,(ex & 0XFF0000) >> 16,(ex & 0XFF000000) >> 24, 0}; //作用是什么 
	//cv::Size S = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),  
	//	(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT) ); 
	//cv::VideoWriter cap_write;
	//cap_write.open(targetVideo,ex, cap.get(CV_CAP_PROP_FPS),S, true); //打开写入文件，并指定格式
	pedTracking(videoname);
}

int pedTracking(const char* videoname)
{
	cv::VideoCapture cap(videoname);
	if(!cap.isOpened())
		return -1;

	SVMDetector detector;
	detector.loadDetectorVector("mydetectorNew.xml");
	detector.initSymmetryParam(527,531,310,248,530,0.75);

	Tracker tracker = Tracker();//distinguish是在tracker中完成的，

	Manager manager = Manager();

	cv::Mat sourceImage;
	cv::Mat gray;
	int interval = 5;//detector检测调用间隔
	int k = 0;//统计调用间隔
	bool isRequest = true;//检测调用请求
	LockedArea* current,*tmp;//记录当前已经检测得到的行人
	Trackerlet* trackerletlist;//tracker向manager提交trackerlet列表
	Trackerlet* correctTrackerlet;//manager向tracker反馈修正结果

	while(cap.read(sourceImage))
	{
		std::cout<<std::endl;
		std::cout<<"NEXT PERIOD:"<<std::endl;
		//检测调用过程，完成行人检测过程
		if( k > interval || isRequest )//两种情况调用：周期调用，响应请求
		{
			cv::cvtColor(sourceImage,gray,CV_BGR2GRAY);
			//cv::imshow("frame",sourceImage);
			detector.detectBaseOnSymmetry(gray);
			current = detector.getResultRect();//获得行人检测结果，
			tmp = current->next;
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
			isRequest = false;
		}
		//对tracker内容进行更新
		isRequest = tracker.update(sourceImage);

		imshow("sourceImage",sourceImage);
		//cap_write<<sourceImage;
		if(!isRequest)//当前tracklet更新成功，可以进行tracklet管理过程
		{
			trackerletlist = tracker.getTrackerlist();
			manager.setTrackerletList(trackerletlist);
			//之后应当是根据传递trackerlet进行判定过程，判定哪个trackerlet属于目标trackerlet，或者说
			//向tracker确定，跟踪目标
			if(!manager.dicision())
			{
				correctTrackerlet = manager.correct();
				tracker.correctTarget(correctTrackerlet);
			}
		} 
		char key = cv::waitKey(3);
		if(key == 27)
			break;
		if(key == 32)
		{
			while(cv::waitKey(3) != 32);
		}
		k++;

		tracker.clearList();
	}
	cap.release();
	cv::waitKey(0);
	return 0;
}