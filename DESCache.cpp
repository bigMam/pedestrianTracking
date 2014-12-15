#include "DESCache.h"

DESCache::DESCache()
{
    count1 = count2 = count4 = 0;
}
DESCache::~DESCache()
{
	pixData.clear();
}

//这里的目的是生成一个计算模板，目标能够对于任意输入图像快速计算其对应特征值，并不一定是LBP,针对其他内容也是可以进行计算的，这点很关键
//
void DESCache::init()
{
	winSize = Size(32,32);
	cellSize = Size(8,8);//cell大小（8*8）

	int i, j;
	//int nbins = 16;//这里需要在这里进行定义么，这是在真正的特征计算中才会进行考虑的，所以在这里并不是必须的
	int rawBlockSize = winSize.width * winSize.height;//block内像素个数

	ncells = Size(winSize.width/cellSize.width, winSize.height/cellSize.height);//一个block中包含cell个数（4*4）
	//winHistogramSize = ncells.width * ncells.height*nbins;
	//一个block生成特征向量维数（4*4*16 = 256维，每个cell生成一个直方图，每个直方图含有9个bins）
	//同样这里winHistogramSize表示的是特征算子维数，这里进行初始化的过程中是没有必要的，之后再单独进行处理，正常的逻辑应该是这样的
	pixData.resize(rawBlockSize*3);//记录每个像素相关信息，包括在当前图像中的偏移量，贡献直方图位置，贡献权重

	//开始对每个pixel进行统计过程
	count1 = count2 = count4 = 0;
	//遍历扫描窗口内某个block
	//计算单个block中的内所有像素的pixData值
	//对单个block进行区域划分如下：
	//{[A][B] [C][D]}
	//{[E][F] [G][H]}
	//
	//{[I][J] [K][L]}
	//{[M][N] [O][P]}    //参考tornadomeet文章内容

	//对每个像素进行计算
	for( j = 0; j < winSize.width; j++ )//winSize.width == 32
	{
		for( i = 0; i < winSize.height; i++ )//winSize.height == 32
		{
			PixData* data = 0;//新建PixData指针
			float cellX = (j+0.5f)/(cellSize.width) - 0.5f;//cellSize.width == 8
			int icellX0 = cvFloor(cellX);
			int icellX1 = icellX0 + 1;
			cellX -= icellX0;//差值
			//j = [0,3] icellX0 = -1，icellX1 = 0;
			//j = [4,11] icellX0 = 0,icellX1 = 1
			//j = [12,19] icellX0 = 1,icellX1 = 2
			//j = [20.27] icellX0 = 2,icellX1 = 3
			//j = [28,31] icellX0 = 3,icellX1 = 4

			float cellY = (i+0.5f)/(cellSize.height) - 0.5f;
			int icellY0 = cvFloor(cellY);
			int icellY1 = icellY0 + 1;
			cellY -= icellY0;
			//i = [0,3] icellY0 = -1，icellY1 = 0;
			//i = [4,11] icellY0 = 0, icellY1 = 1
			//i = [12,19] icellY0 = 1,icellY1 = 2
			//i = [20.27] icellY0 = 2,icellY1 = 3
			//i = [28,31] icellY0 = 3,icellY1 = 4

			//上述操作完成后，直接根据icellY0、icellY1、icellX0、icellX1可判断当前像素所属位置

			//cellY表示差值
			//ncells(2,2),宽高均为2
			if( (unsigned)icellX0 < (unsigned)ncells.width &&
				(unsigned)icellX1 < (unsigned)ncells.width )
			{
				if( (unsigned)icellY0 < (unsigned)ncells.height &&
					(unsigned)icellY1 < (unsigned)ncells.height )
				{
					//能够对四个cell产生影响的像素：F、G、J、K
					//注意这里的unsigned，这里满足该约束条件的只能是icellX0 == 0;icellY0 == 0
					//当前区域内像素对四个cell值均有影响
					//
					//ncells.height == 2
					//原本直接*nbins可以确定偏移量，修改将*nbins过程后移，当前仅仅确定直方图编号
					data = &pixData[rawBlockSize*2 + (count4++)];//跳过前两类，直接对第三类（4）进行赋值操作
					data->histOfs[0] = (icellX0*ncells.height + icellY0);//
					data->histWeights[0] = (1.f - cellX)*(1.f - cellY);//权重，比较巧妙的计算，节省很多繁琐的过程
					data->histOfs[1] = (icellX1*ncells.height + icellY0);//
					data->histWeights[1] = cellX*(1.f - cellY);
					data->histOfs[2] = (icellX0*ncells.height + icellY1);//
					data->histWeights[2] = (1.f - cellX)*cellY;
					data->histOfs[3] = (icellX1*ncells.height + icellY1);//
					data->histWeights[3] = cellX*cellY;
					//histOfs表示当前像素对哪个直方图产生影响，histWeight表示对直方图产生影响的权重
					//其他依次类推

				}
				else
				{
					//区域B、C、N、O
					data = &pixData[rawBlockSize + (count2++)];
					if( (unsigned)icellY0 < (unsigned)ncells.height )//unsigned(-1) > 2
					{
						//N、O
						icellY1 = icellY0;//icellY1 = 1,原值为2
						cellY = 1.f - cellY;
					}
					data->histOfs[0] = (icellX0*ncells.height + icellY1);
					data->histWeights[0] = (1.f - cellX)*cellY;
					data->histOfs[1] = (icellX1*ncells.height + icellY1);
					data->histWeights[1] = cellX*cellY;
					//设定两类权重
					data->histOfs[2] = data->histOfs[3] = 0;
					data->histWeights[2] = data->histWeights[3] = 0;
				}
			}
			else
			{
				if( (unsigned)icellX0 < (unsigned)ncells.width )//icellX0 == 1
				{
					icellX1 = icellX0;
					cellX = 1.f - cellX;
				}

				if( (unsigned)icellY0 < (unsigned)ncells.height &&
					(unsigned)icellY1 < (unsigned)ncells.height )
				{
					//区域E、H、I、L
					data = &pixData[rawBlockSize + (count2++)];
					data->histOfs[0] = (icellX1*ncells.height + icellY0);
					data->histWeights[0] = cellX*(1.f - cellY);
					data->histOfs[1] = (icellX1*ncells.height + icellY1);
					data->histWeights[1] = cellX*cellY;
					data->histOfs[2] = data->histOfs[3] = 0;
					data->histWeights[2] = data->histWeights[3] = 0;
				}
				else
				{
					//区域A、D、M、P
					data = &pixData[count1++];
					if( (unsigned)icellY0 < (unsigned)ncells.height )
					{
						icellY1 = icellY0;
						cellY = 1.f - cellY;//对特殊情况进行了单独处理
					}
					data->histOfs[0] = (icellX1*ncells.height + icellY1);
					data->histWeights[0] = cellX*cellY;
					//仅对自身所在cell产生影响
					data->histOfs[1] = data->histOfs[2] = data->histOfs[3] = 0;
					data->histWeights[1] = data->histWeights[2] = data->histWeights[3] = 0;
				}
			}//这里完成对当前位置像素影响cell位置及相应的权重计算过程
			data->offset = (winSize.width * i + j);//当前像素在整个win中的偏移量
		}//for循环结束，完成对pixData的赋值工作，明确一个block中每个像素负责的bins，及其贡献权重
	}

	assert( count1 + count2 + count4 == rawBlockSize );//最终保证每个像素均参与处理，其总和应为rawBlockSize
    // defragment pixData//碎片整理，保证连续性，也就是对其进行移动
    for( j = 0; j < count2; j++ )
        pixData[j + count1] = pixData[j + rawBlockSize];
    for( j = 0; j < count4; j++ )
        pixData[j + count1 + count2] = pixData[j + rawBlockSize*2];
    count2 += count1;
    count4 += count2;//记录各自的结束位置

	//对pixData进行遍历，查看到底得到了怎样的内容
	//const PixData* _pixData = &pixData[0];//获得pixData的指针
	//int k, C1 = count1, C2 = count2, C4 = count4;
	//for(k = 0; k < C1; k++)
	//{
	//	const PixData& pk = _pixData[k];
	//	std::cout<<"offset:"<<pk.offset;
	//	std::cout<<" histOfs:"<<pk.histOfs[0];
	//	std::cout<<" weight:"<<pk.histWeights[0]<<std::endl;

	//}
	//for(;k < C2; k++)
	//{
	//	const PixData& pk = _pixData[k];
	//	std::cout<<"offset: "<<pk.offset;

	//	std::cout<<" histOfs0:"<<pk.histOfs[0];
	//	std::cout<<" weight0:"<<pk.histWeights[0]<<" ";

	//	std::cout<<" histOfs1:"<<pk.histOfs[1];
	//	std::cout<<" weight1:"<<pk.histWeights[1]<<std::endl;
	//}
	//for(;k < C4;k++)
	//{
	//	const PixData& pk = _pixData[k];
	//	std::cout<<"offset: "<<pk.offset;

	//	std::cout<<"histOfs0: "<<pk.histOfs[0];
	//	std::cout<<"weight0: "<<pk.histWeights[0]<<" ";

	//	std::cout<<"histOfs1: "<<pk.histOfs[1];
	//	std::cout<<"weight1: "<<pk.histWeights[1]<<" ";

	//	std::cout<<"histOfs2: "<<pk.histOfs[2];
	//	std::cout<<"weight2: "<<pk.histWeights[2]<<" ";
	//
	//	std::cout<<"histOfs3: "<<pk.histOfs[3];
	//	std::cout<<"weight3: "<<pk.histWeights[3]<<std::endl;
	//}
}