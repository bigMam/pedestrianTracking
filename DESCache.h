#pragma once
#include <opencv2\core\core.hpp>
#include <iostream>

using namespace cv;
//目标是得到一个计算模板，针对固定尺寸窗口（32*32），对不同原图，计算相应的特征描述
//得到的特征描述，维度为4*4*x，x表示当前直方图的维度，可变
struct PixData//存储某个像素内容,1个PixData结构体1个像素点的数据
{
	size_t offset;//LBPofs，表示偏移量，用于直接定位像素位置
	int histOfs[4];//histOfs[]//这里指的是当前pixel所贡献cell产生直方图起始位置的偏移量 （贡献cell最多有4个）
	float histWeights[4];//histWeight[]贡献权重？？
};
class DESCache
{
public:
	DESCache();
    virtual ~DESCache();

	virtual void init();//对当前尺寸的计算得到cell内像素，贡献直方图位置及相应权重公式

	vector<PixData> pixData;//存数所有像素的数据
	Size winSize;//winSize,扫描窗口大小；
	Size cellSize;
    Size ncells;//当前窗口中cell的个数
    int count1, count2, count4;//统计一个窗口中不同类型像素的个数
};