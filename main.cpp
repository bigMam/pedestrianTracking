#include <iostream>
#include "mySVM.h"
#include "SymmetryProcess.h"

using namespace cv;
int main()
{
	const char* filename =  "D:\\ImageDataSets\\TestSamples\\image1202.jpg";
	const char* videoname = "D:\\ImageDataSets\\trackingSamples\\MVI_2683_08_save.avi";

	SVMDetector detector;
	detector.loadDetectorVector("mydetectorNew.xml");
	detector.initSymmetryParam(527,531,310,248,530,1.2);
	VideoCapture cap(videoname);
	if(!cap.isOpened())
		return -1;
	cv::Mat sourceImage;
	cv::Mat gray;

	LockedArea* current,*tmp;//记录当前已经检测得到的行人

	while(cap.read(sourceImage))
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
		imshow("sourceImage",sourceImage);
		if(cv::waitKey(1) == 27)
			break;
	}
	cv::waitKey(0);
	std::cout<<"hello world"<<std::endl;
	return 0;
}