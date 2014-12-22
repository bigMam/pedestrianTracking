#include "featureExtractor.h"

//分别定义边缘判定系数
const float verCof[] = {1, -1, 1, -1};
const float horCof[] = {1, 1, -1, -1};
const float invCof[] = {1.414f, 0, 0, -1.414f};//45度对角线
const float diaCof[] = {0, 1.414f, -1.414f, 0};//135度对角线
const float nodCof[] = {1, -1, -1, 1};

void FeatureExtractor::initCache()
{
	cache.init();
}
//获得当前窗口内的直方图
const float* FeatureExtractor::getBlockHistogram(float* buf,cv::Mat& img,int nbins)
{
	//cv::resize(img,img,cv::Size(32,32));
	assert(img.cols == 32 && img.rows == 32);

	winHistogramSize =  cache.ncells.width * cache.ncells.height * nbins;//确定当前提取特征向量维度

	float* blockHist = buf;//得到指向直方图位置的指针

	int k, C1 = cache.count1, C2 = cache.count2, C4 = cache.count4;

	for( k = 0; k < winHistogramSize; k++ )
        blockHist[k] = 0.f;//对当前直方图进行初始化操作，初始化为0.f


	const PixData* _pixData = &cache.pixData[0];//获得pixData的指针
	const uchar* lbpPtr = img.ptr<uchar>(0);

	//pixData的存储方式是连续存放的[...C1...C2...C4],所以可以经由k值依次读取，可以完成对一个block中所有像素的遍历
    //先对影响个数为1的像素进行统计，也就是四个角的区域
    for( k = 0; k < C1; k++ )
    {
        const PixData& pk = _pixData[k];
        const uchar* h = lbpPtr + pk.offset;
		float w = pk.histWeights[0];

        int h0 = h[0];//
        float* hist = blockHist + pk.histOfs[0] * nbins;//确定当前像素影响的cell
        float t0 = hist[h0] + w;//累加，对应不同bin值
        hist[h0] = t0;//对影响cell的对应的直方图进行赋值
    }
	for( ; k < C2; k++ )//类似计算
    {
        const PixData& pk = _pixData[k];
        const uchar* h = lbpPtr + pk.offset;
		int h0 = h[0];

        float* hist = blockHist + pk.histOfs[0] * nbins;
        float w = pk.histWeights[0];
        float t0 = hist[h0] + w;
        hist[h0] = t0; 

        hist = blockHist + pk.histOfs[1] * nbins;
        w = pk.histWeights[1];
        t0 = hist[h0] + w;
        hist[h0] = t0;
    }
	for( ; k < C4; k++ )//类似计算
    {
        const PixData& pk = _pixData[k];
        const uchar* h = lbpPtr + pk.offset;
		int h0 = h[0];

        float* hist = blockHist + pk.histOfs[0] * nbins;
        float w = pk.histWeights[0];
        float t0 = hist[h0] + w;
        hist[h0] = t0; 

        hist = blockHist + pk.histOfs[1] * nbins;
        w = pk.histWeights[1];
        t0 = hist[h0] + w;
        hist[h0] = t0;

		hist = blockHist + pk.histOfs[2] * nbins;
        w = pk.histWeights[2];
        t0 = hist[h0] + w;
        hist[h0] = t0;

		hist = blockHist + pk.histOfs[3] * nbins;
        w = pk.histWeights[3];
        t0 = hist[h0] + w;
        hist[h0] = t0;
    }
	normalizeBlockHistogram(blockHist);//对生成的blockHist进行归一化处理

	return blockHist;
}
void FeatureExtractor::normalizeBlockHistogram(float* _hist) const
{
    float* hist = &_hist[0];
    size_t i, sz = winHistogramSize;//blockHistogramSize表示直方图所含维数
    float sum = 0;
    for( i = 0; i < sz; i++ )
        sum += hist[i]*hist[i];//平方和？
    float scale = 1.f/(std::sqrt(sum)+sz*0.1f), thresh = 0.2;
    //获得变换系数，及最大阈值
    for( i = 0, sum = 0; i < sz; i++ )
    {
        hist[i] = std::min(hist[i]*scale, thresh);//在第一次的基础上继续求解平方和
        sum += hist[i]*hist[i];
    }

    scale = 1.f/(std::sqrt(sum)+1e-3f);

    for( i = 0; i < sz; i++ )
        hist[i] *= scale;//直接乘以系数，得到最终的归一化结果
}


/**************************提取HSV颜色空间特征****************************/
void FeatureExtractor::HSVExtractor(const cv::Mat& src,blockFeature& feature)
{
	Mat  hsv;
	cvtColor(src, hsv, CV_RGB2HSV);

	int hueChannel = 0;
	const int hueSize = 180;
	float hranges[] = { 0, 180 };
	const float *hueRange[] = { hranges };
	calcHist( &hsv, 1, &hueChannel, Mat(), // do not use mask
		feature.hueHist, 1, &hueSize,hueRange,
		true, // the histogram is uniform
		false );
	cv::normalize(feature.hueHist,feature.hueHist,1.0,NORM_MINMAX);


	int satChannel = 1;
	const int satSize = 180;
	float sranges[] = { 0, 256 };
	const float *satRange[] = { sranges };
	calcHist( &hsv, 1, &satChannel, Mat(), // do not use mask
		feature.satHist, 1, &satSize,satRange,
		true, // the histogram is uniform
		false );
	cv::normalize(feature.satHist,feature.satHist,1.0,NORM_MINMAX);


	int valChannel = 2;
	const int valSize = 180;
	float vranges[] = { 0, 256 };
	const float *valRange[] = { vranges };
	calcHist( &hsv, 1, &valChannel, Mat(), // do not use mask
		feature.valHist, 1, &valSize,valRange,
		true, // the histogram is uniform
		false );
	cv::normalize(feature.valHist,feature.valHist,1.0,NORM_MINMAX);
	
	//这里留下一个问题，是将内容放在数组中进行计算，
	//还是存在矩阵中进行计算更为方便呢，有待进一步的查看
	//答案是放在矩阵中更为方便，原因是存在一个compareHist函数，可以直接计算两直方图之间的距离
	//包括文献中给出的BHATTACHARYYA距离计算
}


/**************************提取CS_LBP局部二值图特征************************/
template <typename _Tp> static
	void olbp_(InputArray _src, OutputArray _dst) {
		// get matrices
		Mat src = _src.getMat();
		// allocate memory for result
		_dst.create(src.rows-2, src.cols-2, CV_8UC1);
		Mat dst = _dst.getMat();
		// zero the result matrix
		dst.setTo(0);

		//std::cout<<"rows "<<src.rows<<" cols "<<src.cols<<std::endl;
		//std::cout<<"channels "<<src.channels()<<std::endl;
		//getchar();
		// calculate patterns
		for(int i=1;i<src.rows-1;i++) {
			_Tp *pre = src.ptr<_Tp>(i - 1);
			_Tp *cur = src.ptr<_Tp>(i);
			_Tp *post = src.ptr<_Tp>(i + 1);
			_Tp *dest = dst.ptr<_Tp>(i-1);

			for(int j=1;j<src.cols-1;j++) {
				_Tp center = cur[j];
				//cout<<"center"<<(int)center<<"  ";
				unsigned char code = 0;
				code |= (post[j-1] - pre[j+1] > 3) << 3;
				code |= (post[j]   - pre[j]   > 3) << 2;
				code |= (post[j+1] - pre[j-1] > 3) << 1;
				code |= (cur[j+1]  - cur[j-1] > 3) << 0;
				dest[j-1] = (int)code;//simple uniform weight，十七均匀分布？取值为0~15
				//code |= (pre[j-1]  >= center) << 7;  
				//code |= (pre[j]    >= center) << 6;  
				//code |= (pre[j+1]  >= center) << 5;  
				//code |= (cur[j+1]  >= center) << 4;  
				//code |= (post[j+1] >= center) << 3;  
				//code |= (post[j]   >= center) << 2;  
				//code |= (post[j-1] >= center) << 1;  
				//code |= (cur[j-1]  >= center) << 0;  
				//dest[j-1] = code;
				//std::cout<<(int)code<<" ";
				//cout<<(int)code<<endl;
			}
		}
}
void FeatureExtractor::CS_LBPExtractor(const cv::Mat& gray,blockFeature& feature)
{
	
	cv::Mat lbp;
	olbp_<uchar>(gray,lbp);
	//cv::imshow("lbp",lbp);

	int nbins = 16;
	int winHistogramSize = 4 * 4 * nbins;

	feature.cs_lbpFeature.resize(winHistogramSize);
	float *buf = &feature.cs_lbpFeature[0];
	if (cache.count4 == 0)
		cache.init();

	cv::Mat dest;
	cv::resize(lbp,dest,cv::Size(32,32));
	getBlockHistogram(buf,dest,nbins);
}


/**************************提取Canny边缘特征********************************/
void FeatureExtractor::CannyExtractor(const cv::Mat& gray,blockFeature& feature)
{
	
	cv::Mat edge;
	int lower = 40;
	int upper = 40 * 1.5;
	cv::Canny(gray,edge,lower,upper);
	//cv::imshow("edge",edge);
	//cv::normalize(edge,edge,1.0,0.0,NORM_MINMAX);
	for(int i = 0; i < edge.rows; i++)
	{
		uchar* ptr = edge.ptr<uchar>(i);
		for(int j = 0; j < edge.cols; j++)
		{
			if((int)ptr[j] == 255)
			{
				ptr[j] = 1;
			}
		}
		//std::cout<<std::endl;
	}

	int nbins = 2;
	int winHistogramSize = 4 * 4 * nbins;

	feature.cannyFeature.resize(winHistogramSize);
	float *buf = &feature.cannyFeature[0];
	if (cache.count4 == 0)
		cache.init();
	getBlockHistogram(buf,edge,nbins);
}


/**************************提取垂直水平方向一阶求导特征**********************/
void FeatureExtractor::horVerDerExtractor(const cv::Mat& gray,blockFeature& feature)
{

	cv::Mat sobelX;
	//参数为：源图像，结果图像，图像深度，x方向阶数，y方向阶数，核的大小，尺度因子，增加的值
	Sobel(gray,sobelX,CV_8U,1,0,CV_SCHARR,0.4,128);
	//imshow("X方向Sobel结果",sobelX);

	cv::Mat sobelY;
	Sobel(gray,sobelY,CV_8U,0,1,CV_SCHARR,0.4,128);
	//imshow("Y方向Sobel结果",sobelY);

	int Channel = 0;
	const int histSize = 180;
	float ranges[] = { 0, 256 };
	const float *Range[] = { ranges };
	calcHist( &sobelX, 1, &Channel, Mat(), // do not use mask
		feature.horDerHist, 1, &histSize,Range,
		true, // the histogram is uniform
		false );

	cv::normalize(feature.horDerHist,feature.horDerHist,1.0,NORM_MINMAX);
	

	calcHist( &sobelY, 1, &Channel, Mat(), // do not use mask
		feature.verDerHist, 1, &histSize,Range,
		true, // the histogram is uniform
		false );
	cv::normalize(feature.verDerHist,feature.verDerHist,1.0,NORM_MINMAX);
}


/**************************提取EHD边缘直方图特征******************************/
//用于存储边缘判断系数
typedef struct _filter
{
	float LT;
	float RT;
	float LB;
	float RB;
}edgeFilter;
//利用4像素方差来判断当前cell是否为单调
bool isMonotone(cv::Mat& cell)
{
	assert(cell.cols == 2 && cell.rows == 2 );

	uchar* ptr = cell.ptr<uchar>(0);
	uchar* ptr2 = cell.ptr<uchar>(1);
	float r = (ptr[0] + ptr[1] + ptr2[0] + ptr2[1]) / 4.0f;

	float delta = ((ptr[0] - r) * (ptr[0] - r)  + (ptr[1] - r) * (ptr[1] - r) + 
		(ptr2[0] - r) * (ptr2[0] - r) + (ptr2[1] - r) * (ptr2[1] - r)) / 4.0f;
	if (delta < 15)
		return true;
	else
		return false;
}
bool isNoDirection(cv::Mat& cell)
{
	return false;
}
//返回值0~5,分别表示垂直、水平、45度、135度、无规则边缘、单调无边缘信息
int judgeEdgeType(cv::Mat& cell)
{
	assert(cell.cols == 2 && cell.rows == 2 );

	//当前cell是否为单调cell，利用方差进行判断
	if (isMonotone(cell))
	{	
		return 5;
	}
	if(isNoDirection(cell))
	{
		return 4;
	}
	int thresholdOfEdge = 11;

	float m[5] = {0,0,0,0,0};

	//通过计算系数分别当前单元格各类型的响应值
	for(int i = 0; i < 2; i++)
	{
		uchar* ptr = cell.ptr<uchar>(i);
		for(int j = 0; j < 2 ; j++)
		{
			int grayValue = ptr[j];
			m[0] += grayValue * verCof[i * 2 + j];
			m[1] += grayValue * horCof[i * 2 + j];
			m[2] += grayValue * invCof[i * 2 + j];
			m[3] += grayValue * diaCof[i * 2 + j];
			m[4] += grayValue * nodCof[i * 2 + j];
		}
	}

	float max = 0;
	int index = 0;
	//寻找m数组中的最大值
	for(int i = 0; i < 5; i++)
	{
		if(max < abs(m[i]))
		{
			max = abs(m[i]);
			index = i;
		}
	}
	if(max < thresholdOfEdge)//没用类型响应值超过设定阈值
	{
		return 5;
	}
	else
	{
		return index;
	}
}
void FeatureExtractor::EHDExtarctor(const cv::Mat& gray,blockFeature& feature)
{
	int winWidth = gray.cols;
	int winHeight = gray.rows;
	int cellWidth = 2;//单元格尺寸
	int cellHeight = 2;
	cv::Size numOfCells = cv::Size(winWidth / cellWidth,winHeight / cellHeight);//当前窗口内单元格个数

	int count = 0;
	for(int i = 0; i < 5;i ++)
	{
		feature.EHD[i] = 0;
	}
	//下面要解决的问题便是如何快速完成统计过程，
	for(int i = 0; i < numOfCells.width; i++)
	{
		for(int j = 0; j < numOfCells.height; j++)
		{
			cv::Mat cell = gray(cv::Rect(i * cellWidth,j * cellHeight,2,2));
			int type = judgeEdgeType(cell);
			if (type != 5)
			{
				feature.EHD[type]++;
			}
			else
			{
				count++;
			}
		}
	}
	float numOfCell = numOfCells.width * numOfCells.height;
	for(int i = 0; i < 5; i ++)
	{
		feature.EHD[i] = feature.EHD[i] / numOfCell;
	}
}
//统一所有特征提取过程
void FeatureExtractor::computeFeature(const cv::Mat& src,blockFeature& feature)
{
	cv::Mat dest;
	cv::resize(src,dest,cv::Size(32,32));
	cv::Mat gray;
	cv::cvtColor(dest,gray,CV_RGB2GRAY);

	HSVExtractor(dest,feature);
	CS_LBPExtractor(gray,feature);
	CannyExtractor(gray,feature);
	horVerDerExtractor(gray,feature);
	EHDExtarctor(gray,feature);
}

