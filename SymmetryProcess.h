#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>

struct LockedArea
{
	int topLeftX;
	int topLeftY;
	int width;
	int height;
	LockedArea* next;
};
class SymmetryProcess
{
public:
	SymmetryProcess(void);
	SymmetryProcess(float ax,float ay,float u0,float v0,float f);//内参，焦距也可以看做是内参的一部分，毕竟是不会发生变化的
	SymmetryProcess(float ax,float ay,float u0,float v0,float f,float theta,float high);
	SymmetryProcess(float ax,float ay,float u0,float v0,float f,float theta,float high,
				int Rx,int Ry,float aspectRatio,float minHigh, float maxHigh);

	~SymmetryProcess(void);

	void setResolutionRatio(int Rx,int Ry);//设定分辨率
	void setExternalParam(float theta,float high);//设定外参
	void setAspectRatio(float aspectRatio);//设定行人高宽比
	void setRealObjectHigh(float minHigh, float maxHigh);//设定行人最小最大高度

	void initParam();//利用内外参数，对程序中所需参数进行计算，

	void loadImage(const char* filename);//加载待处理图像
	void loadImage(cv::Mat& image);//加载图像

	void cannyProc();//由图像得到对应的边缘图像
	void AddScanLines();//对边缘图像添加扫描线，

	float computeWeight(int x,int center,int width);
	void showScanningWindows();//绘制设定扫描窗口大小
	void computeSymmetryCurve();//沿扫描线计算对称值曲线
	void plotCurve();//根据矩阵绘制对称值曲线

	void eliminate();//消除不满足约束条件的对称值
	void statisticNew();//新的统计过程，统计底端位置，及纵向累加值
	void extractPeaks();//针对对称值曲线提取峰值
	void lockPedestrianArea();//根据对称值峰值信息，对行人区域进行锁定锁定之后，symmetryProcess类的任务已经完成了，之后的
						//的工作，应当交由其他的类来完成，保证类的功能的专一性
	LockedArea* getAreaInfo();

	void getArea();//得到不同模板各自的扫描区域，，由底端边缘的统计信息求得，这里的边缘信心是经过简单过滤得到的，
	void plotArea();//在原始图像中将扫描区域描绘出来，，把实验结果清晰的展示出来
	void plotBox(cv::Mat &targetImage,int topLeftX,int topLeftY,int width,int height);//给定左右边界及底端，在原始图像中进行绘制
	
	//void clusterPeaks();//对峰值进行聚类操作，得到聚类中心
	//void estimateBoundBox();//根据聚类中心估计边界框，
	//因为是逐个扫描区域进行处理，所以需要扫描区域信息进行遍历，
	void scanningAndVerify();//逐个扫描区域进行扫描，确定是否内部有行人存在
	void getTemplateMinMax(int linesNum,int &minTemplate,int &maxTemplate);//经由linesNum（1~12）确定当前扫描区域选用模板尺寸范围，高度范围
	int GetMin(int a, int b, int c, int d, int e);
	void productModel(const char* filename);//产生边缘响应值匹配模板


	/*目前可以完成的工作包括：对当前程序进行整理，统一变量
	对程序进行改进
	1，对扫描线的左右边界进行扩展到到img边界位置；
	2，估计对称值的最小阈值，用来对区域进行消除操作
	3，关于区域的合并、扩展工作如何合理的完成
	4，根据处理后的扫描区域明确最终扫描区域的上下界范围，也就是确定最终的待扫描区域
	5，明确最终得到的待扫描区域，是为多个区域的集合，并对不同位置的区域采取相应的扩展/收缩工作，再交由分类器进行分类
	将分类结果反推得到原始图像中的行人区域，
	还是有很多工作要完成的，这里使用参数已知相机获取图像，并参与计算过程
	，，最近两天时间完成上述工作，必须自己给自己抓紧了，，
	思路是将区域摘取出来然后进行resize，检测，在还原回原始图像中对的位置，这样可行？会不会增加计算的时间复杂度
	*/
private:
	//摄像机基本参数,在对摄像机进行标定之后，可以直接输入
	float ax;
	float ay;
	float u0;
	float v0;
	float f;//焦距

	float theta;//摄像机俯仰角
	float high;//摄像机距离地面高度
	int Rx,Ry;//当前摄像机分辨率

	float aspectRatio;//统计得到，行人宽高比
	float maxRealHobj,minRealHobj;//实际行人高度取值范围[1.5,1.9]

	//由给定参数计算得到的变量内容,还是存在思路不清晰的地方
	//扫描工作是在尺寸变化后的（240*320）图像上展开的
	float Wcoff,Hcoff;//尺寸比例变化系数
	float aspectRatioNew;//在320*240图像中行人高宽比，
	int groundLine;//地平面消失位置，原始图像中
	int startPos;//扫描线起点，扫描图像中
	int endPos;//扫描线终点,扫描图像中

	int highLayer[5][2];//分层记录startPos及endPos处对应的最大最小高度，每层的划分依据是实际身高[1.5,1.9]五层
						//每层包含三个数据，当前层的最小高度，最大高度，及递增值
	//这些内容可以稍微靠后一些，整体框架并没有完全做好，只能说是完成了边缘对称性检测部分，
	//后面的分类器对目标进行确认的工作还没有展开呢
	int interval;//表示扫描线的间隔
	

	cv::Mat sourceImage;//读入原始图像 
	cv::Mat destImage;//将原始图像改变为320*240大小扫描图像
	cv::Mat edgeImage;//利用扫描图像得到边缘信息图像
	cv::Mat responseModel;//扫描窗口边缘响应值匹配模板
	
	int modelWidth;
	int modelHeight;

	static const int linesNum = 12;//扫描线个数,确定为12
	static const int scanningWidth = 320;//扫描线宽度确定为320
	int bottomInfo[scanningWidth][2];//保存320列对称值信息。第一个数据表示当前列的下边缘位置[1,12]，第二个数据表示当前列的对称值累加和
	int peakInfo[linesNum * 2];//记录对称值累加值 峰值所在位置，
	int symmetryCurve[linesNum][scanningWidth + 1];//最后一个元素，存放当前行中最大值


	LockedArea* lockedPedArea;//用于保存提取出来的原始图像中行人矩形框信息,链表形式，因为不知道最终提取区域数目

	struct AreaInfo
	{
		int linesNum;
		int startPos;
		int endPos;
		AreaInfo* next;
	};
	AreaInfo scanningArea[linesNum];//存储每条扫描线上所有可能扫描区域的信息

};


