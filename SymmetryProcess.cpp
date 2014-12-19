#include "SymmetryProcess.h"
#include <iostream>
#include <math.h>
using namespace cv;
using namespace std;
SymmetryProcess::SymmetryProcess(void)
{
	//linesNum = 12;//[120~230]
}

SymmetryProcess::SymmetryProcess(float ax,float ay,float u0,float v0,float f)//内参，焦距也可以看做是内参的一部分，毕竟是不会发生变化的
{
	this->ax = ax;
	this->ay = ay;
	this->u0 = u0;
	this->v0 = v0;
	this->f = f;
}
SymmetryProcess::SymmetryProcess(float ax,float ay,float u0,float v0,float f,float theta,float high)
{
	new (this)SymmetryProcess(ax,ay,u0,v0,f);
	setExternalParam(theta,high);
}
SymmetryProcess::SymmetryProcess(float ax,float ay,float u0,float v0,float f,float theta,float high,
	int Rx,int Ry,float aspectRatio,float minHigh,float maxHigh)
{
	new (this)SymmetryProcess(ax,ay,u0,v0,f,theta,high);
	setResolutionRatio(Rx,Ry);
	setAspectRatio(aspectRatio);
	setRealObjectHigh(minHigh,maxHigh);
}
void SymmetryProcess::setResolutionRatio(int Rx,int Ry)//设定分辨率
{
	this->Rx = Rx;
	this->Ry = Ry;
}
void SymmetryProcess::setExternalParam(float theta,float high)//设定外参
{
	this->theta = theta;
	this->high = high;
}
void SymmetryProcess::setAspectRatio(float aspectRatio)//设定行人高宽比
{
	this->aspectRatio = aspectRatio;
}

void SymmetryProcess::setRealObjectHigh(float minHigh,float maxHigh)
{
	this->minRealHobj = minHigh;
	this->maxRealHobj = maxHigh;
}

SymmetryProcess::~SymmetryProcess(void)
{
}




void SymmetryProcess::initParam()
{

	//////////暂时不用如此这般的处理，仅仅是知道能够这样计算就可以了，这些参数的求解还有待进一步的实验，暂时搁置///////////
	Wcoff = 320.0 / Rx;
	Hcoff = 240.0 / Ry;
	aspectRatioNew = aspectRatio * (Wcoff / Hcoff);

	groundLine = v0 - ay * tanf(theta);
	int groundLineN = groundLine * Hcoff;
	startPos = groundLineN + 5;
	endPos = 235;
	interval = (endPos - startPos)/11;

	int startPosS = startPos / Hcoff;
	int endPosS = endPos / Hcoff;//由扫描线位置反推在原始img图像中的位置
	
	float startHigh,endHigh;
	float startZw = (ay * high) / (startPosS - groundLine);//由反推得到的扫描线位置确定实际距离，再由距离推导高度
	float endZw = (ay * high) / (endPosS - groundLine);
	float realHigh = 1.5;

	for(int i = 0; i < 5; i++)
	{
		startHigh = (f / startZw) * realHigh;
		endHigh = (f / endZw) * realHigh;
		highLayer[i][0] = startHigh * Hcoff;//在扫描图像中扫描窗口的最小高度，startPos
		highLayer[i][1] = endHigh * Hcoff;//扫描图像中的扫描窗口的最大高度，endPos
		//highLayer[i][2] = (highLayer[i][1] - highLayer[i][0]) / (linesNum - 1);
		realHigh = realHigh + 0.1;
	}
	//这些均是在尺寸变化后图像中的参数，原本这些参数是经由原始img图像参数及变化系数共同决定的

	//aspectRatioNew = 0.3f;//设定宽高比
	//maxHigh = 180;//单位为像素,
	//minHigh = 60;
	//startPos = 120;
	//endPos = 230;
	//interval = (endPos - startPos)/11;//10

	lockedPedArea = new LockedArea(); 
	lockedPedArea->topLeftX = 0;
	lockedPedArea->topLeftY = 0;
	lockedPedArea->width = 0;
	lockedPedArea->height = 0;
	lockedPedArea->next = NULL;



}
void SymmetryProcess::loadImage(const char* filename)
{
	sourceImage = cv::imread(filename,0);
	int rows = sourceImage.rows;
	int cols = sourceImage.cols;
	cout<<"rows:"<<rows;
	cout<<"cols:"<<cols;
	cout<<"channels:"<<sourceImage.channels()<<endl;
	Wcoff = 320.0/cols;
	Hcoff = 240.0/rows;
	groundLine = rows / 2;//粗粒度处理，有待细化，暂时可用
}

void SymmetryProcess::loadImage(cv::Mat& image)
{
	sourceImage = image;//使用图像头指向相同内存空间
}

void SymmetryProcess::cannyProc()
{
	clock_t start,end;
	resize(sourceImage,destImage,Size(320,240));//>>>O_O<<<<这里存在问题，直接将原始图像转变为320*240，对图像在一定程度上造成形变，
	
	//需要考虑进来。或者换一个思路，不直接强行改变，而是按照比例进行缩放
	start = clock();
	GaussianBlur(destImage, destImage, Size(7,7), 1.5, 1.5);
	int low = 40;
	double high = low / 0.4;
	
	Canny(destImage, edgeImage, low, high, 3);//>>>O_O<<<<这里有一点文章可做，怎样选择最优的阈值，使得前景与背景尽可能多的分离开来
	
	end = clock();
	std::cout<<"边缘提取耗时："<< end - start <<std::endl;
	//保证行人最大可能的凸显出来，有文章可做
	//imshow("edge",edgeImage);
}
void SymmetryProcess::AddScanLines()
{
	//这里的处理是在edgeImage上进行划线，按照给定参数将扫描线在edgeImage上描绘出来
	//这里的描绘是使得，结果看上去更加直观，并不是说必须的操作
	int count = 0;
	for(int i = startPos; count < linesNum;)
	{
		uchar *edgePtr = edgeImage.ptr<uchar>(i);
		for(int j = 0; j < edgeImage.cols; j++)//从画线开始对扫描线左右进行扩展
		{
			if (edgePtr[j] != 255)
				edgePtr[j] = 180;
		}
		i = i + interval;
		count++;
	}
	imshow("ScanLine",edgeImage);
}
// 显示不同扫描线上设定的扫描窗口范围
void SymmetryProcess::showScanningWindows()
{
	float increase = (highLayer[4][1] - highLayer[4][0]) / (linesNum - 1);//得到窗口高度递增值
	cv::Mat showWindows = cv::Mat::zeros(240,320,CV_8UC1);
	//在中间位置进行绘制，看看效果如何
	for(int i = 0; i <linesNum; i++)
	{
		int winHigh = highLayer[4][0] + i * increase;
		float winWidth = winHigh * aspectRatioNew;//计算得到当前扫描窗口大小
		int rowPos = startPos + i * interval;//得到当前扫描线位置
		plotBox(showWindows,110,rowPos-winHigh,winWidth,winHigh);
	}
	imshow("scanning Window",showWindows);
}

//这里利用x 距离中心的距离，计算权重，考虑使用高斯函数的简化模式
//尝试更为平滑的过渡。当前并没有使用高斯函数，而是分段函数激进型处理的，
//该机制的效果，只能做到这一步了，除非有更为优秀的方法，仅仅是表面上的缝缝补补并不能有很大的提高
//暂且先这样用着，开始对其他问题进行解决，对扫描区域进行锁定
float SymmetryProcess::computeWeight(int x,int center,int width)
{
	int distance = std::abs(x - center);
	float weight;

	//float delta = width / 6.0;
	//weight = 1 / delta * sqrt(2 * 3.14) * exp(- distance * distance / (2 * delta * delta));
	if(distance < width / 6.0)
	{
		weight = 1 - 0.5 * distance / (width / 2.0) * 3;
	}
	else if(distance < width / 3.0)
	{
		weight = 0.75 - 0.25 * distance / (width / 2.0) * 3;
	}
	else
	{
		weight = 0.5 - 0.125 * distance / ( width / 2.0) * 3;
	}
	return weight;
}

//>>>O_O<<<<这里的扫描过程，肯定还需要进一步修改，不光是这里，还有其他的很多地方，当前的任务是，将整个流程过一遍
//这里将沿着扫描线分别计算对称值，这是本程序的重点部分
//存在的主要问题是，如何确定每个扫描线上的扫描窗口大小，
//假设扫描线为目标的中心位置，在该前提条件下，反推扫描窗口的大小，设定行人宽高比为0.4，行人高度为1.5~1.8（这个数据有用么）
//这里的确定与摄像机本身参数有关，需要推算求得
//在没有推算的情形下，先简单假设，对实验进行验证
//如何进行假设？设定宽高比，高度最大及最小，中间依次递减，
void SymmetryProcess::computeSymmetryCurve()
{
	//对对称值数组进行初始化操作
	for(int i = 0 ; i < linesNum; i++)
	{
		for(int j = 0; j < scanningWidth + 1; j++)
		{
			symmetryCurve[i][j] = 0;
		}
	}
	float increase = (highLayer[4][1] - highLayer[4][0]) / (linesNum - 1);//得到高度递增
	uchar* scanLinePtr;
	uchar* scanWinTP;
	//uchar *modelPtr = edgeImage.ptr<uchar>(0);//指向匹配模板的起始位置
	//对每条扫描线进行处理，从上往下，依次进行 
	for(int i = 0; i < linesNum; i++)
	{
		int winHigh = highLayer[4][0] + i * increase;
		int winWidth = winHigh * aspectRatioNew;//计算得到当前扫描窗口大小

		//已知扫描窗口尺寸的情形下，要确定扫描窗口内元素，与响应模板的一一对应关系,需要注意的是不能发生越界
		//float widthTR = modelWidth / winWidth;
		//float heightTR = (float)modelHeight / winHigh;

		int rowPos = startPos + i * interval;//得到当前扫描线位置
		if(rowPos - winHigh > 0)//当前行扫描窗口顶端未越界
		{
			scanLinePtr = edgeImage.ptr<uchar>(rowPos);//获得指向扫描线所在行的指针
			int maxVal = 0;//记录每行中的最大值
			//计算每个可计算点的对称性，这里的可计算点指的是，该店对应的扫描窗口在图像范围之内，并没有发生越界，
			for(int j = 0; j < scanningWidth ; j = j + 1)//总计扫描宽度为scanningWidth，为已经之前设定好的
			{
				if(j - winWidth / 2.0 > 0 && j + winWidth / 2.0 < 320)//当前位置，对应扫描窗口左右边界未越界
				{
					int maxDistance = (int)sqrt((double)(winHigh * winHigh + (winWidth/2.0) * (winWidth/2.0)));
					int symVal = 0;//初始化每个中心点对称值为零
					int oldVal = symVal;//oldVal用于判断对称值保持不变的连续次数
					int continued = 0;//记录SymVal值连续未发生变化的次数这里的连续未变化是否可行呢？
					//在未发生越界的情况下，对扫描窗口内关于中心的对称值进行计算，利用论文中给出的公式
					//如果连续30行Symval值未发生变化，则可判断当前未出现行人，存在的若干对称值仅为干扰项
					for(int y = winHigh ;y > 0; y--)//自下而上进行计算
					{
						scanWinTP = scanLinePtr - (winHigh - y)*scanningWidth + j - winWidth/2;//再次犯了上次的错误，不应该
						for(int x = 1;x < winWidth / 2.0; x++)
						{
							//这里也是出现了四层for循环有没有可能降低一层或更多呢？留给优化
							//继续对对称值进行修正，这里由于人体本身的不规则性，选择当前点，对应点包含附近位置来
							//计算对称性，附近暂时先使用x轴方向，表临近

							if(scanWinTP[x] == 255 && (scanWinTP[x] == scanWinTP[winWidth - x - 1]|| 
								scanWinTP[x] == scanWinTP[winWidth - x]||scanWinTP[ x] == scanWinTP[winWidth - x + 1]))
							{
								//symVal = symVal + 1;
								//这里对对称值的求解进行改进，不再是简单记作1，
								//______而是一个以距离中线距离x及当前窗口面积Area作为参数的值
								//symVal = symVal + 1;
								//symVal = symVal + 100.0 * (winWidth / 2.0 - x) / (winWidth/2.0) * 100.0 *((float)(y)/winHigh);
								//______考虑距离底端中点距离，比例值与距离长反比
								//symVal = symVal + maxDistance / sqrt((double)((winHigh - y)*(winHigh - y) + x * x));

								//继续对称值的求解过程，尝试分区域进行讨论权值，近似的进行区间划分，之后进行分别进行求解
								//按照y值进行划分，粗粒度划分，不同的位置设定不同的边缘权重设定

								if( y < winHigh /7)//头部
								{
									int center = winWidth * 2 / 3.0;
									int distance = std::abs(x - center);//x 距离权重最大值点距离，权重与距离成反比
									float weight = computeWeight(x,center,winWidth);
									symVal = symVal + 1000.0 * weight * 100.0 *(((float)y) / winHigh);
								}
								else if(y < (winHigh * 4.0) / 7)//上半身
								{
									int center = 0;
									int distance = std::abs(x - center);//x 距离权重最大值点距离，权重与距离成反比
									float weight = computeWeight(x,center,winWidth);
									symVal = symVal + 1000.0 * weight * 100.0 *(((float)y) / winHigh);
								}
								else //下半身
								{
									int center = winWidth * 2 / 3.0;
									int distance = std::abs(x - center);//x 距离权重最大值点距离，权重与距离成反比
									float weight = computeWeight(x,center,winWidth);
									symVal = symVal + 1000.0 * weight * 100.0 *(((float)y) / winHigh);
								}
								//目前来看方法三，效果最优，但这就是最好的了么？为什么高斯函数不能够使用

								//这里将最后的*100 / winHigh移至累加结束之后，减少计算量
							}
							else if(scanWinTP[j - x] != scanWinTP[j + x])
							{
								//symVal = symVal -  1;//00.0 * x / (winWidth/2.0);//* 10000.0 / (winHigh * winHigh);
							}
						}
						int k = interval*0.8;
						if((winHigh - y) == k && symVal == 0)//自下而上连续若干行都没有对称值出现，说明不包含行人，有点莽撞
							break;

						//这里的判断还有待进一步确定，当前的做法并不正确，没有达到预期的效果
						if(symVal == oldVal)
						{
							continued++;
							if(continued == 40)//暂定为30行，经验主义，没有什么特别的依据
							{
								symVal = 0;
								break;
							}
						}
						else
						{
							continued = 0;
						}
						oldVal = symVal;
					}
					symVal = symVal * 100.0 / (winHigh);
					if(symVal > 0)//这里在不考虑非对称产生的影响的情况下，总为正值
						symmetryCurve[i][j] = symVal;//j 与symmetryCurve中的下表是意义对应的
					else
						symmetryCurve[i][j] = 0;
					if (symVal > maxVal) 
						maxVal = symVal;
				}
			}
			symmetryCurve[i][320] = maxVal;//将该行最大值进行保存，属于历史遗留问题，当下已经没有用处了？关于阈值的设定问题
		}
	}
}
//根据矩阵SymmetryCurve绘制曲线，曲线的绘制也不是必须的，不过是使得过渡更自然，结果更清晰
void SymmetryProcess::plotCurve()
{
	cv::Mat curve = cv::Mat::zeros(240,320,CV_8UC1);//得到一个全零矩阵
	//对每条曲线进行手动绘制，若为零，则直接绘制，最高不超过4个间隔
	int maxHigh = 4 * interval;
	float maxVal = 0;
	for(int i = 0; i < linesNum;i++)
	{
		if(maxVal < symmetryCurve[i][320])
			maxVal = symmetryCurve[i][320];
	}
	//获取所有对称值的最大值
	float ratio = maxHigh / maxVal;//获得每个val值，在高度显示中的比例值

	for(int i = 0; i < linesNum; i++)
	{
		int rowPos = startPos + i * interval;//得到当前扫描线位置
		uchar *edgePtr = curve.ptr<uchar>(rowPos);//获得指向当前所描线指针
		for(int j = 0; j < scanningWidth; j++)//逐点进行绘制
		{
			uchar *edgeP = edgePtr + j;
			int currVal = symmetryCurve[i][j];
			int high = currVal * ratio;//获得当前值的像素的相对高度
			edgeP = edgeP - 320*high;
			edgeP[0] = 255;//对指定单元进行赋值
		}
	}
	imshow("curve",curve);
}

//消除不满足约束条件的行，当前的消除依据主要包括，
//孤立行，指的是当前扫描线上的连续对称值个数小于5个，且在左右10个像素内没有对称值，这里设定的值，是暂时的，需要进行测试
//纵向对称值累加值小于定值，这里的定值，每个扫描线对应一个值，小于该定值，说明当前的对称值的求解一定不是由行人边缘产生的
//目前想到的内容就只有这么多，先将其实现吧
void SymmetryProcess::eliminate()
{
	int minWidth = 5;//连续对称值最小宽度
	int maxBlank = 10;//允许最大空白宽度
	//两者能否同时完成呢，以减少循环次数。思路不够清晰，就写不出来哈
	for(int i = 0; i < linesNum; i++)//遍历每一行，依次进行消除工作
	{
		int maxVal = symmetryCurve[i][320];//得到每行的最大值

		int leftBorder = 0;
		int rightBorder = 0;//连续对称值的左右边界
		int leftBlankLength = 0;
		int rightBlankLength = 0;
		int continuesLength = 0;//连续对称值左右空白长度，用来判定是否将该连续段清楚

		bool isFromBlank = true;//表示当前元素的前一个元素属性,前一个元素不是blank就是symmVal，初始设定为true，虚拟一个blank
		bool isLeft = true;//标识当前空白相对连续对称部分位置

		//尽量做到一次遍历
		for(int j = 0; j < scanningWidth; j++)
		{
			int symmetryVal = symmetryCurve[i][j];
			if(symmetryVal < maxVal * 0.2)//这里的设定其实是不合理的，因为可能整个扫描线上都不存在合理的对称值
				//但如此操作，扫描线上就一定会保留部分；或者另外一种情况，所有对称值都合理，这样会误删掉部分
				//不论是否为零，只要小于定值，就一律按作空白处理
			{
				symmetryCurve[i][j] = 0;//置零
				if(isFromBlank)
				{
					if(isLeft)//如果为左空白，默认为左空白
						leftBlankLength++;
					else
						rightBlankLength++;
				}
				else//表明前一个单元为symmVal
				{
					rightBorder = j - 1;
					isLeft = false;//一次判断过程中 一左一右，所以在记录过程中只有一次置为false
					rightBlankLength++;
				}
			}
			else
			{
				if(isFromBlank)
				{
					if(isLeft)
					{
						leftBorder = j;
						continuesLength++;
						isFromBlank = false;//当前已经进入symm范围，下次再进行判断是，该值当然要设定为false了
					}
					else//从右侧空白遭遇新的symmVal
					{
						isFromBlank = false;//isFromBlank == false，isLeft == False右侧空白遭遇symmVal的情形下出现
									//还有一种可能会出现，从symmVal刚刚进入右空白时，此时，isLeft == false；isFromBlank == false
					}
				}
				else
				{
					continuesLength++;
				}
			}

			//解读能否完成所有记录工作
			//对记录信息进行判断，什么时候能进行判断，在连续部分的后一个空白区间内进行判断
			if(!isLeft)//表明为右空白
			{
				if(continuesLength > minWidth)//表明不需要进行消除工作，再次进行初始化工作
				{
					leftBorder = 0;
					rightBorder = 0;
					leftBlankLength = rightBlankLength;
					rightBlankLength = 0;
					continuesLength = 0;
					isFromBlank = true;
					isLeft = true;
				}
				else//需要进一步通过左右空白进行判断
				{
					if (leftBlankLength < maxBlank)
					{
						leftBorder = 0;
						rightBorder = 0;
						leftBlankLength = rightBlankLength;
						rightBlankLength = 0;
						continuesLength = 0;
						isFromBlank = true;
						isLeft = true;
					}
					else//当前需要对右侧空白长度进行判断，一旦超过则进行删除操作
					{
						if(rightBlankLength > maxBlank)//需要进行删除
						{
							for(int k = leftBorder; k < rightBorder + 1; k++)
							{
								symmetryCurve[i][k] = 0;//置零
							}
							leftBorder = 0;
							rightBorder = 0;
							leftBlankLength = leftBlankLength + continuesLength + rightBlankLength;
							rightBlankLength = 0;
							continuesLength = 0;
							isFromBlank = true;
							isLeft = true;
						}
						else if(isFromBlank)
						{
							//不作处理，循环继续
						}
						else
						{
							if(!(j - rightBorder == 1))//表明不是刚刚进入右侧空白
							{
								//右空白遭遇symmVal，但是右空白长度不足，直接再次进行初始化
								leftBorder = j;
								rightBorder = 0;
								leftBlankLength = rightBlankLength;
								rightBlankLength = 0;
								continuesLength = 1;
								isFromBlank = false;
								isLeft = true;
							}
						}
					}
				}
			}
		}
	}

	//绘制处理后的图像
	float maxVal = 0;
	for(int i = 0; i < linesNum;i++)
	{
		if(maxVal < symmetryCurve[i][320])
			maxVal = symmetryCurve[i][320];
	}
	float ratio = 200 / maxVal;

	//对对称值矩阵进行简单模糊处理
	int tmp[12][scanningWidth];
	for(int i = 0; i < 12;i++)
	{
		tmp[i][0] = symmetryCurve[i][0];
		for(int j = 1; j < scanningWidth - 1; j++)
		{
			tmp[i][j] = (symmetryCurve[i][j -1] + symmetryCurve[i][j]*2 + symmetryCurve[i][j+1])/4;
		}
		tmp[i][scanningWidth - 1] = symmetryCurve[i][scanningWidth - 1];
	}
	cv::Mat target = Mat(Size(320,120),CV_8U);
	//对新建矩阵进行赋值操作
	for(int i = 0;i < linesNum; i++)
	{
		int rowPos = i * 10;
		uchar *edgePtr = target.ptr<uchar>(rowPos);
		//对首行进行赋值
		for(int j = 0;j < scanningWidth;j++)
		{
			if(bottomInfo[j][0] != 0)
				edgePtr[j] = tmp[i][j] * ratio;//使用模糊后的数据
			else
				edgePtr[j] = 0;
		}
		//对剩余九行进行直接拷贝
		for(int k = 1;k < 10; k++)
		{
			target.row(rowPos).copyTo(target.row(rowPos + k));
		}
	}
	imshow("after statistics target",target);
}

//新的统计过程，统计底端位置，及纵向累加值
void SymmetryProcess::statisticNew()
{
	//分别记录每个扫描线作为底端的列的累加最大值，暂时保留，不做其他处理
	int thresholdMax[linesNum];
	//下面两个for循环可以进行合并
	for(int j = 0; j < scanningWidth; j++)//对symmetryCurve中的每一列自下而上进行统计，记录其第一个不为零的数据位置
	{
		bottomInfo[j][0] = 0;
		for(int i = linesNum - 1; i > 1;i--)
		{
			if(symmetryCurve[i][j] != 0)
			{
				bottomInfo[j][0] = (i + 1);
				break;
			}
		}
	}
	//求解每列累加对称值之和
	for(int j = 0; j < scanningWidth; j++)
	{
		bottomInfo[j][1] = 0;
		for(int i = 0; i < bottomInfo[j][0]; i++)
		{
			bottomInfo[j][1] = bottomInfo[j][1] + symmetryCurve[i][j];
		}
		//记录每个底端对应的对称值最大取值情况，目的是后面的消除工作
		//这里暂时先放在这里，不作处理。
		if(bottomInfo[j][0] != 0)//当底端bottomInfo[j][0]不为零时，比较累加值与当前记录值的大小关系
		{
			int bottom = bottomInfo[j][0] - 1;//bottomInfo[j][0]的取值范围是[1,linesNum]，因而这里需要进行减一操作
			if(thresholdMax[bottom] < bottomInfo[j][1])
			{
				thresholdMax[bottom] = bottomInfo[j][1];
			}
		}
		if(j > 1)
		{
			bottomInfo[j-1][1] = (bottomInfo[j -2][1] + bottomInfo[j-1][1]*2 + bottomInfo[j][1])/4;
		}
	}

	//for(int j = 0; j < scanningWidth; j++)//对symmetryCurve中的每一列自下而上进行统计，记录其第一个不为零的数据位置
	//{
	//	bottomInfo[j][0] = 0;
	// 	bool isFirst = true;
	//	for(int i = linesNum - 1; i > 2;i--)
	//	{
	//		if(symmetryCurve[i][j] != 0 && isFirst)
	//		{
	//			bottomInfo[j][0] = (i + 1);
	//			isFirst == false;
	//		}
	//		bottomInfo[j][1] = bottomInfo[j][1] + symmetryCurve[i][j];
	//	}
	//	if(bottomInfo[j][0] != 0)//当底端bottomInfo[j][0]不为零时，比较累加值与当前记录值的大小关系
	//	{
	//		int bottom = bottomInfo[j][0] - 1;//bottomInfo[j][0]的取值范围是[1,linesNum]，因而这里需要进行减一操作
	//		if(thresholdMax[bottom] < bottomInfo[j][1])
	//		{
	//			thresholdMax[bottom] = bottomInfo[j][1];
	//		}
	//	}
	//}

	//当前包含信息，每列的底端位置，每列的底端累加值，
	//不用考虑各个底端在不同的扫描线上，最终的累加效果是一样的，所以当前只需要在bottomInfo[i][1]中寻找峰值就可以了


	////对峰值进行绘制
	//int maxPlotLength = 50;
	//int maxSymVal = thresholdMax[10];
	//float ra = (float)maxPlotLength / maxSymVal;
	//for(int i = 0; i < scanningWidth; i++)
	//{
	//	int symmVal = bottomInfo[i][1];
	//	int currentHigh = symmVal * ra;
	//	for(int k = 0; k < currentHigh; k++)
	//	{
	//		std::cout<<".";
	//	}
	//	std::cout<<endl;
	//}
}

//提取峰值信息,
//包括提取峰值和合并临近峰值的工作。考虑非极大值抑制
void SymmetryProcess::extractPeaks()
{
	for(int j = 0; j < linesNum * 2; j++)
	{
		peakInfo[j] = 0;//对峰值信息进行初始化
	}
	
	int recordSub[scanningWidth / 2];
	int k = 0;//利用k值对reocrdSub进行记录遍历
	int p,q;//标记两次非极大抑制求解峰值在recordSub中的边界
	int i = 1;
	while( i + 1 < scanningWidth)
	{
		if(bottomInfo[i][1] > bottomInfo[i + 1][1])
		{
			if(bottomInfo[i][1] > bottomInfo[i - 1][1] || bottomInfo[i][1] == bottomInfo[i - 1][1])
			{
				//cout<<"peak1 at "<<i<<endl;
				recordSub[k++] = i;
			}
		}
		else
		{
			i = i + 1;
			while(i + 1 < scanningWidth && (bottomInfo[i][1] < bottomInfo[i + 1][1] || bottomInfo[i][1] == bottomInfo[i + 1][1]) )
			{
				i = i + 1;
			}
			if(i + 1 < scanningWidth)
			{
				//peak at i;
				//cout<<"peak1 at "<<i<<endl;
				recordSub[k++] = i;
			}
		}
		i = i + 2;
	}
	recordSub[k++] = -1;
	p = k;
	//再次对已求得峰值利用非极大抑制，寻求真正峰值
	i = 1;
	while(i + 1 < p - 1)
	{
		if(bottomInfo[recordSub[i]][1] > bottomInfo[recordSub[i + 1]][1])
		{
			if(bottomInfo[recordSub[i]][1] > bottomInfo[recordSub[i -1]][1] || 
				bottomInfo[recordSub[i]][1] == bottomInfo[recordSub[i -1]][1])
			{
				//cout<<"peak2 at "<<recordSub[i]<<endl;
				recordSub[k++] = recordSub[i];
			}
		}
		else
		{
			i = i + 1;
			while(recordSub[i + 1] != -1 &&
				(bottomInfo[recordSub[i]][1] < bottomInfo[recordSub[i + 1]][1] ||
				bottomInfo[recordSub[i]][1] == bottomInfo[recordSub[i + 1]][1]))
			{
				i = i + 1;
			}
			if(i + 1 < p - 1)
			{
				//peak at i;
				//cout<<"peak2 at "<<recordSub[i]<<endl;
				recordSub[k++] = recordSub[i];
			}
		}
		i = i + 2;
	}
	q = k;
	//第二次非极大值抑制求解得到的为真正峰值，在recordSub中的范围是[p,q],两次利用非极大抑制，干得漂亮
	k = p;
	for(int j = 0; k < q;j++,k++)
	{
		peakInfo[j] = recordSub[k];//将峰值信息进行保存
	}
}
//第一步将区域在原始图像中框出
//第二步将框内的图像提取出来进行模板匹配，这个过程因为涉及到 
//图像的resize，可能会比较耗时，，老规矩先做出来，看看效果 如何之后再进行进一步的处理
void SymmetryProcess::lockPedestrianArea()

{
	//这里是直接进行初始化操作，如果是连续处理的话，还需要对lockedPedArea链表信息进行释放，切记切记
	LockedArea* current = lockedPedArea->next;
	LockedArea* tmp;
	while(current)
	{
		tmp = current->next;
		delete current;
		current = tmp;
	}
	lockedPedArea->next = NULL;
	int location,scanLineNum;
	float increase;
	int bottomPos,bottomPosS;
	int winHigh,winWidth,winHighS,winWidthS,topLeftX,topLeftY;

	for(int i = 0; peakInfo[i]!= 0;i++)
	{
		location = peakInfo[i];
		scanLineNum = bottomInfo[location][0];//得到峰值所在底端位置[1,12]
		//每个位置分别绘制五层边框
		//for(int j = 0; j < 5;j++)
		//{
		//	increase = (float)(highLayer[j][1] - highLayer[j][0]) / (linesNum - 1);
		//	winHigh = highLayer[j][0] + (scanLineNum - 1) * increase;//这里的高度为最大高度1.9，对应的高度，如何确定不同高度对应值呢
		//							//初步设定范围是[1.5,1.9]共5个范围，只要是高度确定了，对应的宽度也就确定了
		//	winWidth = winHigh * aspectRatioNew;
		//	winHighS = winHigh / Hcoff;
		//	winWidthS = winWidth / Wcoff;

		//	bottomPos = startPos + (scanLineNum - 1) * interval;
		//	bottomPosS = bottomPos / Hcoff;

		//	topLeftX = location / Wcoff - winWidthS / 2;
		//	topLeftY = bottomPosS - winHighS;

		//	LockedArea* tmpArea = new LockedArea();
		//	tmpArea->topLeftX = topLeftX;
		//	tmpArea->topLeftY = topLeftY;
		//	tmpArea->width = winWidthS;
		//	tmpArea->height = winHighS;
		//	tmpArea->next = lockedPedArea->next;
		//	lockedPedArea->next = tmpArea;
		//	plotBox(sourceImage,topLeftX,topLeftY + 3*i ,winWidthS,winHighS);
		//}

		increase = (float)(highLayer[3][1] - highLayer[3][0]) / (linesNum - 1);
		winHigh = highLayer[2][0] + (scanLineNum - 1) * increase;//这里的高度为最大高度1.9，对应的高度，如何确定不同高度对应值呢
		//初步设定范围是[1.5,1.9]共5个范围，只要是高度确定了，对应的宽度也就确定了
		winWidth = winHigh * aspectRatioNew;
		winHighS = winHigh / Hcoff;
		winWidthS = winWidth / Wcoff;

		bottomPos = startPos + (scanLineNum - 1) * interval;
		bottomPosS = bottomPos / Hcoff;

		topLeftX = location / Wcoff - winWidthS / 2;
		topLeftY = bottomPosS - winHighS;

		LockedArea* tmpArea = new LockedArea();
		tmpArea->topLeftX = topLeftX;
		tmpArea->topLeftY = topLeftY;
		tmpArea->width = winWidthS;
		tmpArea->height = winHighS;
		tmpArea->next = lockedPedArea->next;
		lockedPedArea->next = tmpArea;
		plotBox(sourceImage,topLeftX,topLeftY ,winWidthS,winHighS);

	}
	//完成对提取行人区域的保存，下一步是在对其进行resize，并计算得到特征描述算子进行分类匹配
	//LockedArea* p = lockedPedArea->next;
	//while(p!= NULL)
	//{
	//	cout<<p->height<<" ";
	//	p = p->next;
	//}
	//std::cout<<std::endl;
	imshow("sourceImage0",sourceImage);//显示效果不错，下一步，将行人区域提取出来，进行特征值计算及后期的匹配
	
}


LockedArea* SymmetryProcess::getAreaInfo()
{
	return lockedPedArea;
}
//这里的getArea完成了删除孤立列及合并相邻列的操作
void SymmetryProcess::getArea()
{
	//得到各个模块的扫描区域，简单来讲就是，经由底端边缘信息，bottomInfo数组进行统计得到
	//可以明确的一点是，这里得到的最终结果是扫描区域，而不是特定的扫描窗口的定位，有可能会进行一定程度的合并。
	//现在的问题,1、利用怎样的数据结构进行存储
	//2、采取怎样的策略进行区域求解。最简单的思路就是直接进行统计
	
	//对区域信息进行初始化操作
	for(int i = 0; i < linesNum; i++)
	{
		scanningArea[i].linesNum = 0;
		scanningArea[i].startPos = 0;
		scanningArea[i].endPos = 0;
		scanningArea[i].next = NULL;
	}

	for(int i = scanningWidth - 1; i > 0; )
	{
		while(bottomInfo[i][0] == 0 && i > 0)
		{
			i--;
		}//略过底端为0的所有列，结束时，当前列底端不为零
		int firstVal = bottomInfo[i][0];//记录当前底端值[1,12]
		AreaInfo* area = new AreaInfo();
		area->linesNum = bottomInfo[i][0];
		area->endPos = i;
		i--;
		while(bottomInfo[i][0] == firstVal && i > 0)
		{
			i--;
		}//结束时表示当前列的底端值不再为firstVal，可能为零，也可能为另外一个值

		area->startPos = i + 1;
		area->next = scanningArea[firstVal - 1].next;
		scanningArea[firstVal - 1].next = area;
	}
	//显示
	for(int i = 0; i < linesNum; i++)
	{
		AreaInfo* p = scanningArea[i].next;
		if (p != NULL)
		{
			cout<<"linesNum "<<p->linesNum<<":";
		}
		while(p != NULL)
		{
			cout<<p->startPos<<","<<p->endPos<<" ";
			p = p ->next;
		}
		cout<<endl;
	}
	//统计结束之后，对邻近部分进行合并，同时过滤掉仅有一列的元素,剔除单列元素较为简单，麻烦的是如何进行合并，

	//需要设定列数少于多少进行删除，距离小于多少进行合并
	//首先是判断能否进行合并，能合并，则合并之，其次在没有进行合并的情况下，再判断是否需要进行清除
	
	int minCols = 3;//设定最少列数
	int maxDistance = 8;//设定两个非连续列的最大可合并距离，这里的设定暂时没有什么依据，直观感受而已
	AreaInfo* preNode;//指向当前处理的两个节点之前的节点
	AreaInfo* first;//指向当前处理的第一个节点
	AreaInfo* second;//指向当前处理的第二个节点
	for(int i = 0; i < linesNum; i++)
	{
		preNode = &scanningArea[i];
		AreaInfo* p = scanningArea[i].next;
		//需要进行统一操作
		while( p != NULL)
		{
			first = p;
			p = p->next;
			if (p != NULL)//表明存在second，则进行判断，合并或者其他
			{
				//首先对是否可以进行合并进行判断
				second = p;
				if((second->startPos - first->endPos) + 1 < maxDistance)//可以进行合并操作
				{
					first->endPos = second->endPos;
					first->next = second->next;
					delete second;
					p = first;
				}
				//对是否进行删除进行判断，对两个底端分别进行判断
				else if((first->endPos - first->startPos + 1) < minCols && (second->endPos - second->startPos + 1) < minCols)
				{
					//同时删除两个节点
					preNode->next = second->next;
					delete first;
					delete second;
					p = preNode->next;

				}
				else if((first->endPos - first->startPos + 1) < minCols)
				{
					//删除第一个节点
					preNode->next = second;
					delete first;
					p = second;
				}
				else if((second->endPos - second->startPos + 1) < minCols)
				{
					//删除第二个节点
					first->next = second->next;
					delete second;
					p = first;
				}
				else
				{
					//两个节点均保存下来
					p = second;
				}

			}
			else//仅存在一个first，则直接进行minCols判断
			{
				if((first->endPos - first->startPos + 1) < minCols)
				{
					//删除first节点
					preNode->next = first->next;
					delete first;
					p = preNode->next;
				}
			}
		}
	}
	
	//显示
	for(int i = 0; i < linesNum; i++)
	{
		AreaInfo* p = scanningArea[i].next;
		if (p != NULL)
		{
			cout<<"linesNum "<<p->linesNum<<":";
		}
		while(p != NULL)
		{
			cout<<p->startPos<<","<<p->endPos<<" ";
			p = p ->next;
		}
		cout<<endl;
	}
}
//在原始图像中绘制指定位置边框
void SymmetryProcess::plotBox(Mat &targetImage,int topLeftX,int topLeftY,int width,int height)
{
	//在原始图像sourceImage中绘制扫描区域边框
	//uchar* topPtr = targetImage.ptr<uchar>(topLeftY);
	//uchar* bottomPtr = targetImage.ptr<uchar>(topLeftY + height);
	//for(int i = 0; i < width;i++)
	//{
	//	topPtr[topLeftX + i] = 255;
	//	bottomPtr[topLeftX + i] = 255;
	//}

	//for(int j = 0; j < height;j++)
	//{
	//	topPtr[topLeftX + j * targetImage.cols] = 255;
	//	topPtr[topLeftX + width + j * targetImage.cols] = 255;
	//}
	Rect rect = Rect(topLeftX,topLeftY,width,height);
	cv::rectangle(sourceImage,rect,Scalar(255,255,255),1);
	
}

void SymmetryProcess::plotArea()
{
	float increase = (highLayer[4][1] - highLayer[4][0]) / (linesNum - 1);
	for(int i = 0; i < linesNum; i++)
	{
		AreaInfo* p = scanningArea[i].next;
		int winHigh = highLayer[4][0] + i * increase;
		int winWidth = winHigh * aspectRatioNew;
		int winHighS = winHigh / Hcoff;
		int winWidthS = winWidth / Wcoff;
		if (p != NULL)
		{
			cout<<"linesNum"<<p->linesNum<<":";
		}
		while(p != NULL)
		{
			
			int left = p->startPos;
			int leftS = left / Wcoff;
			
			int bottom = startPos + (p->linesNum - 1) * interval;
			int bottomS = bottom / Hcoff;
			//由leftS，rightS，bottomS共同确定扫描区域大小

			plotBox(sourceImage,leftS,bottomS - winHighS,winWidthS,winHighS);
			p = p ->next;
		}
		cout<<endl;
	}

	imshow("plotAre",sourceImage);
}
//这个函数的作用是什么呢？
void SymmetryProcess::getTemplateMinMax(int num,int &minTemplate,int &maxTemplate)
{
	int bottom = startPos + (num - 1) * interval;
	int bottomS = bottom / Hcoff;
	float Zw = (ay * high) /(bottomS - groundLine);
	minTemplate = (f / Zw) * minRealHobj;
	maxTemplate = (f / Zw) * maxRealHobj;
}

void scanning(){}

//对确定的扫描区域进行确认操作，
void SymmetryProcess::scanningAndVerify()
{
	int num = 0;
	int minTemplate;
	int maxTemplate;
	for(int i = 0; i < linesNum; i++)
	{
		AreaInfo* p = scanningArea[i].next;
		if (p != NULL)
		{
			num = p->linesNum;
			getTemplateMinMax(num,minTemplate,maxTemplate);
		}
		while(p != NULL)
		{
			int left = p->startPos;
			int right = p->endPos;
			int leftS = left / Wcoff;
			int rightS = right / Wcoff;
			int bottom = startPos + (num - 1) * interval;
			int bottomS = bottom / Hcoff;
			//由leftS，rightS，bottomS共同确定扫描区域大小
			scanning();
			p = p ->next;
		}
		cout<<endl;
	}
}
int SymmetryProcess::GetMin(int a, int b, int c, int d, int e)
{
	int t = (a < b ? a : b) < c ? (a < b ? a : b) : c;
	return ((t < d ? t : d) < e ? (t < d ? t : d) : e);
}
//计算图像到设定边缘（非刚体，自行绘制）的距离，利用该距离信息，作为边缘响应值计算权值
void SymmetryProcess::productModel(const char* filename)
{
	responseModel = cv::imread(filename,0);
	int cols = responseModel.cols;//54
	int rows = responseModel.rows;//128
	modelWidth = cols;
	modelHeight = rows;

	int p0,p1,p2,p3,p4;
	int step = responseModel.step;
	int max = 0;
	cv::Mat destImage = cv::Mat(360,150,CV_8UC1);
	for(int i = 0; i < rows;i++)
	{
		uchar* ptr = responseModel.ptr<uchar>(i);
		//uchar* destPtr = destImage.ptr<uchar>(i);
		for(int j = 0; j < cols;j++)
		{
			uchar* pTemp = ptr;
			int min;
			if((i > 0 && i < rows - 1) && (j > 0 && j < cols - 1))
			{
				
					p0 = pTemp[j];
					p4 = pTemp[j - 1] + 3;
					pTemp = ptr - step;
					p1 = pTemp[j - 1] + 4;
					p2 = pTemp[j] + 3;
					p3 = pTemp[j + 1] + 4;
					min = GetMin(p0,p1,p2,p3,p4);
					//destPtr[j] = min;
					ptr[j] = std::min(255,min);
				
			}
			else
			{
				//destPtr[j] = 0;
				ptr[j] = 255;
			}
		}
	}
	for(int i = rows - 1; i > -1; i--)
	{
		uchar* ptr = responseModel.ptr<uchar>(i);
		//uchar* destPtr = destImage.ptr<uchar>(i);
		for(int j = cols - 1;j > -1; j--)
		{
			uchar* pTemp = ptr;
			int min;
			if((i > 0 && i < rows - 1) && (j > 0 && j < cols - 1))
			{
			
					p0 = pTemp[j];
					p1 = pTemp[j + 1] + 3;
					pTemp = ptr + step;
					p2 = pTemp[j - 1] + 4;
					p3 = pTemp[j] + 3;
					p4 = pTemp[j + 1] + 4;
					min = GetMin(p0,p1,p2,p3,p4);
					//destPtr[j] = min;
					ptr[j] = min;
					max = max < min ? min : max; 
				
			}
			else
			{
				//destPtr[j] = 0;
				ptr[j] = 255;
			}
		}
	}
	for(int i = 0; i < rows; i++)
	{
		uchar* ptr = responseModel.ptr<uchar>(i);
		for(int j = 0; j < cols; j++)
		{
			//float temp = destPtr[j];
			//destPtr[j] = 255 * temp / max;
			if(ptr[j] != 255)
			{
				float temp = ptr[j];
				ptr[j] = 150 * ((max - temp) / max);
			}
			else
			{
				ptr[j] = 0;
			}
		}
	}
	cv::imshow("responsemodel",responseModel);
}


