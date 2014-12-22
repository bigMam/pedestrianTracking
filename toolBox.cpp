#include <opencv2/highgui/highgui.hpp>
#include <iostream>
int videoCut(const char* sourceVideo,const char* targetVideo,int startSec,int endSec)
{
	cv::VideoCapture cap(sourceVideo);
	if(!cap.isOpened())
		return -1;
	int ex=static_cast<int>(cap.get(CV_CAP_PROP_FOURCC)); 
	char EXT[] = {ex & 0XFF , (ex & 0XFF00) >> 8,(ex & 0XFF0000) >> 16,(ex & 0XFF000000) >> 24, 0}; //作用是什么 
	cv::Size S = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),  
        (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT) ); 

	double fps = cap.get(CV_CAP_PROP_FPS);//帧频数，即每秒包含帧个数
	int startFrameNum = startSec * fps;
	int endFrameNum = endSec * fps;
	std::cout<<startFrameNum<<" "<<endFrameNum<<std::endl;
	cv::VideoWriter cap_write;
	cap_write.open(targetVideo,ex, cap.get(CV_CAP_PROP_FPS),S, true); //打开写入文件，并指定格式
	cv::Mat frame;
	int count = 0;//对当前帧进行计数，并将保存选定片段
	while(cap.read(frame))
	{
		cv::imshow("frame",frame);
		if(count > startFrameNum && count < endFrameNum)
			cap_write<<frame;
		else if(count > endFrameNum)
			break;

		char key = cv::waitKey(3);
		if(key == 27)
			break;
		if(key == 32)
		{
			while(cv::waitKey(3) != 32);
		}
		count++;
	}
	cv::waitKey();
	cap.release();
	return 0;
}