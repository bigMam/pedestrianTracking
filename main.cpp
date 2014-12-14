#include  "mySVM.h"
#include "SymmetryProcess.h"
#include "featureExtractor.h"
#include "myTracker.h"

using namespace cv;
int main()
{
	const char* filename =  "D:\\ImageDataSets\\TestSamples\\image1202.jpg";
	const char* videoname = "D:\\ImageDataSets\\trackingSamples\\MVI_2683_08_save.avi";
	SVMDetector detector;
	detector.loadDetectorVector("mydetectorNew.xml");
	detector.initSymmetryParam(527,531,310,248,530,1.2);

	Tracker tracker = Tracker();

	VideoCapture cap(videoname);
	if(!cap.isOpened())
		return -1;
	cv::Mat sourceImage;
	cv::Mat gray;
	int interval = 10;//detector检测调用间隔
	int k = 0;//统计调用间隔
	bool isRequest = false;
	LockedArea* current,*tmp;//记录当前已经检测得到的行人

	while(cap.read(sourceImage))
	{
		//调用检测过程
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
			isRequest = false;
		}
		//imshow("sourceImage",sourceImage);

		//在tracker中进行跟踪，已知sourceImage内容，
		//这里将针对新的sourceImage产生新的tarcklet，同时对其中的特征进行更新
		if(!tracker.update(sourceImage))
		{
			isRequest = true;//表明当前未能够及时更新tracklet，需要重新检测行人
		}
		else
		{
			//将更新得到tracklet用于后续的manager，这个后续再继续进行吧
			//git测试代码
			std::cout<<"github test"<<std::endl;
		}
		imshow("sourceImage",sourceImage);
		if(cv::waitKey(1) == 27)
			break;
		k++;
	}
	cv::waitKey(0);
	return 0;
}
