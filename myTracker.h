//完成功能，由底层提供行人矩形框架，产生可靠tracklet，并提交给上层manager
#include "SymmetryProcess.h"//lockedArea
#include "featureExtractor.h"//blockFeature
#include "opencv2/video/tracking.hpp"

typedef struct _trackerlet
{
	int trackerletID;
	int topLeftX;
	int topLeftY;
	int width;
	int Height;
	blockFeature featureSet;//每个trackerlet都有对应的特征提取，用于之后进行前后差异性对比
	_trackerlet* next;

}Trackerlet;

class Tracker
{
	const static int stateNum = 4;//状态矩阵
	const static int measureNum = 2;//测量矩阵，
	cv::Mat state; // (x,y,dX,dY)
	cv::Mat processNoise;
	cv::Mat measurement;
	FeatureExtractor extractor;
	cv::KalmanFilter KF;//先设定一个kalman滤波器，看一下，如何进行操作

	LockedArea *lockedPedArea;//检测得到行人存在区域
	Trackerlet *trackerletHead;//也是链表的形式？
public:
	Tracker();
	void setLoackedPedArea(LockedArea *result);
	bool update(cv::Mat &souceImage);//对之前tracklet进行更新，及产生新的tracklet，用于管理,如果更新失败，则设定request
};