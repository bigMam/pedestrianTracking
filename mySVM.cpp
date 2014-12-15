#include "mySVM.h"

using namespace std;
using namespace cv;



SVMDetector::SVMDetector()
{
	svm_classifier = MySVM();
	hog = HOGDescriptor(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);//初始化HOG描述算子，用于之后的计算
	detectorDim = 3781;
	//hog_cache.init(&hog);//这里完成对blockData，及pixData的计算，之后的操作中只需要简单进行调用就可以了
	//之所以能够进行这样操作，原因在于，已经假设待处理图像的尺寸为64*128
}

void SVMDetector::loadImage(const char* filename)
{
	sourceImage = cv::imread(filename,0);//加载待检测的图像
}

void SVMDetector::loadImage(cv::Mat& image)
{
	image.copyTo(sourceImage);
}

int SVMDetector::computeDetectorVector()
{

	/*************************************************************************************************
	线性SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
	将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个列向量。之后，再该列向量的最后添加一个元素rho。
	如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
	就可以利用你的训练样本训练出来的分类器进行行人检测了。
	***************************************************************************************************/
	if(svm_classifier.get_support_vector_count() == 0)
	{
		std::cout<<"未加载svm分类器"<<std::endl;
		return 0;
	}
	int DescriptorDim = svm_classifier.get_var_count();//特征向量的维数，即HOG描述子的维数
	int supportVectorNum = svm_classifier.get_support_vector_count();//支持向量的个数
	cout<<"支持向量个数："<<supportVectorNum<<endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果

	//将支持向量的数据复制到supportVectorMat矩阵中
	for(int i=0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm_classifier.get_support_vector(i);//返回第i个支持向量的数据指针
		for(int j=0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i,j) = pSVData[j];
		}
	}

	//将alpha向量的数据复制到alphaMat中
	double * pAlphaData = svm_classifier.get_alpha_vector();//返回SVM的决策函数中的alpha向量
	for(int i=0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0,i) = pAlphaData[i];
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？
	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子

	//将resultMat中的数据复制到数组myDetector中
	for(int i=0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0,i));
	}
	//最后添加偏移量rho，得到检测子
	myDetector.push_back(svm_classifier.get_rho());
	cout<<"检测子维数："<<myDetector.size()<<endl;//得到参与计算的detectorVector，之后就是根据
	detectorDim = myDetector.size();

	//classifier的类型，展开相应的计算过程
	//设置HOGDescriptor的检测子
}
void SVMDetector::saveDetectorVector(const char* filename)
{
	FileStorage fs(filename, FileStorage::WRITE);
	cv::Mat mDetector = cv::Mat(myDetector);
	fs<<"detector"<<mDetector;
	fs.release();
}
void SVMDetector::loadDetectorVector(const char* filename)
{
	myDetector.clear();
	FileStorage fs(filename, FileStorage::READ);
	cv::Mat mDetector;
	fs["detector"] >> mDetector;
	float *ptr = mDetector.ptr<float>(0);
	for(int i = 0 ; i < detectorDim; i++)
	{
		myDetector.push_back(ptr[i]);
	}
	fs.release();
}

//计算待检测图像的特征向量值，这里的输入图像统一为64*128格式，
//先暂时使用hog提供的特征向量计算方法进行计算，可行
void SVMDetector::computeDescriptor(cv::Mat& tmpImage,std::vector<float>& descriptors)
{
	hog.compute(tmpImage,descriptors);//直接利用hog自带方法进行检测，存在blockData及pixData的重复计算
}
//利用已经加载的分类器对samples进行分类预测
//先假设使用线性核函数进行处理，如果可行，再进一步进行扩展
//目前来看是简单粗暴了一些，但是不影响功能的实现吧,这里尝试使用poly方法进行预测，看一下效果如何
bool SVMDetector::predict(std::vector<float>& samples,int var_count)
{
	double s = 0;
	int k;
	for(k = 0; k <= var_count - 4; k += 4 )
		s += samples[k]*myDetector[k] + samples[k+1]*myDetector[k+1] +
		samples[k+2]*myDetector[k+2] + samples[k+3]*myDetector[k+3];
	for( ; k < var_count; k++ )
		s += samples[k]*myDetector[k];

	s = s + myDetector[k];
	if(s > 0)
		return true;
	else 
		return false;
}

void SVMDetector::initSymmetryParam()
{
	//SymmetryProcess(float ax,float ay,float u0,float v0,float f,float theta,float high,
	//	         int Rx,int Ry,float aspectRatio,float minHigh, float maxHigh);
	//sp = SymmetryProcess(950,954,528,363,948,0,1.4,1024,768,0.4,1.5,1.9);//这里的参数暂时是这样给定，
											//实际应用中可能需要利用参数给定，这里的参数并不是固定不变的
	sp = SymmetryProcess(527,531,310,248,530,0,1.2,640,480,0.4,1.5,1.9);
	sp.initParam();
}
void SVMDetector::initSymmetryParam(float ax,float ay,float u0,float v0,float f,float h)
{
	sp = SymmetryProcess(ax,ay,u0,v0,f,0.0,h,640,480,0.4,1.5,1.9);
	sp.initParam();
}

void SVMDetector::detectBaseOnSymmetry(cv::Mat &sourceImage)
{
	clock_t start,end;
	start = clock();
	sp.loadImage(sourceImage);//加载待处理图像,需要进行判断，如果不是灰度图像，需要进行转化

	sp.cannyProc();//提取边缘信息
	//sp.AddScanLines();//在指定区域设定扫描线，非必要
	sp.computeSymmetryCurve();//重点，计算扫描线上各像素对称值
	//sp.plotCurve();  //针对每条扫描线绘制对应的对称值曲线图，非必要
	sp.eliminate();  //消除过程最最终的候选区域确定并没用明显的影响，

	sp.statisticNew();//统计扫描线底端位置，及纵向累加值，用于确定候选区域 bottomInfo
	sp.extractPeaks();//二次使用非极大值抑制算法得到纵向累加值的峰值信息 peakInfo
	sp.lockPedestrianArea();//得到预处理结果，若干行人区域，之后就是//这里需要进行改进
							//对原始图像中对应区域进行resize，计算特征向量进行匹配
	end = clock();
	std::cout<<"基于对称性的候选区域确定，耗时："<<end - start << std::endl;
	lockedPedArea = sp.getAreaInfo();//完基于边缘对称性的候选区域的确定，该检测过程还能否进行优化？有待进一步的确定
	//获得原始图像中候选区域，之后的操作是对候选区域逐个进行验证
	LockedArea* post = lockedPedArea;
	LockedArea* current = lockedPedArea->next;

	cv::Mat tmpImage;
	std::vector<float> descriptor;
	start = clock();

	//检测新思路，从小到大依次进行检测，如果检测成功，则该系列后续均不需要继续检测，不一定删除之

	while(current != NULL)
	{
		cv::Mat pedROI = cv::Mat(sourceImage,cv::Rect(current->topLeftX,current->topLeftY,current->width,current->height));
		cv::resize(pedROI,tmpImage,Size(64,128),0,0,INTER_AREA);//对图像进行resize
		computeDescriptor(tmpImage,descriptor);
		if(!predict(descriptor,3780))
		{
			//删除该矩形
			LockedArea* tmp = current;
			post->next = current->next;
			current = current->next;
			delete tmp;
		}else{
			post = current;
			current = current->next;
		}
	}
	end = clock();
	std::cout<<"候选区域验证耗时："<<end - start<<std::endl;
}
LockedArea* SVMDetector::getResultRect()
{
	return lockedPedArea->next;
}

//在进行视频检测之前已经完成的任务包括：svmDetector的载入，对称性检测的初始化工作（如果需要），
int SVMDetector::detectOnVideo(const char* filename)
{
	initSymmetryParam(527,531,310,248,530,1.2);
	VideoCapture cap(filename);
	if(!cap.isOpened())
		return -1;
	namedWindow("frame",1);
	cv::Mat gray;
	while(cap.read(sourceImage))
	{
		cv::cvtColor(sourceImage,gray,CV_BGR2GRAY);
		cv::imshow("frame",sourceImage);
		detectBaseOnSymmetry(gray);

		LockedArea* current = lockedPedArea->next;
		while(current != NULL)
		{
			Rect rect = Rect(current->topLeftX,current->topLeftY,current->width,current->height);
			cv::rectangle(sourceImage,rect,Scalar(0,0,0),1);
			current= current->next;
		}
		imshow("sourceImage",sourceImage);
		if(cv::waitKey(1) == 27)
			break;
	}
	cap.release();
	return 0;
}
