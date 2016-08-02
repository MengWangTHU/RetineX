//#include <iostream>
//#include <opencv2\opencv.hpp>
//#include <Eigen/Dense>
//#pragma comment(lib, "D:\\LIB\\libfftw3-3.lib")
//#pragma comment(lib, "D:\\LIB\\libfftw3f-3.lib")
//#pragma comment(lib, "D:\\LIB\\libfftw3l-3.lib")
//#include <stdio.h>
//#include <stdlib.h>
//#include <fftw3.h>
//
//using namespace cv;
//using namespace std;
//using namespace Eigen;
//
////-----------------------------
////描述：用于Mat的单通数据转换为Matrix的float矩阵
////-----------------------------
//MatrixXd getMatrix(Mat source)
//{
//	MatrixXd result(source.rows, source.cols);
//	uchar* inputImageData1 = source.data;
//	for (int i = 0; i < source.rows; i++)
//	{
//		for (int j = 0; j < source.cols; j++)
//		{
//			int index = i*source.cols + j;
//			double b = inputImageData1[index];
//			result(i, j) = b;
//		}
//	}
//	return result;
//}
//MatrixXd GetGauss(int _sigma, int rows, int cols)
//{
//	MatrixXd result(rows, cols);
//	double sum = 0.0000000;
//	//给x矩阵赋值
//	for (int i = 0; i < rows; i++)
//	{
//		for (int j = 0; j < cols; j++)
//		{
//			double bx = -(double)((cols - 1.00) / 2.00) + (double)j;
//			double by = -(double)((rows - 1.00) / 2.00) + (double)i;
//			double num = -(bx*bx + by*by) / (2 * _sigma*_sigma);
//			result(i, j) = exp(num);
//			sum += result(i, j);
//		}
//	}
//	result = result / (sum + 0.001);
//	return result;
//}
//Mat fft2(Mat src)
//{
//	Mat Fourier = Mat::zeros(src.rows, src.cols, CV_32F);
//	Mat planes[] = { Mat_<float>(src), Mat::zeros(src.size(), CV_32F) }; //定义一个planes数组
//	merge(planes, 2, Fourier);
//	dft(Fourier, Fourier, cv::DftFlags::DFT_COMPLEX_OUTPUT);
//
//	/*vector<Mat> vec;
//	split(Fourier, vec);
//	return vec.at(0);*/
//	return Fourier;
//}
//MatrixXd  fft(MatrixXd sourse)
//{
//	MatrixXd result = sourse;
//	//进行傅立叶变换  利用FFTW库
//	fftw_complex *in, *out;
//	fftw_plan p;
//	//申请一个行数乘以列数的一维向量，并且没列数个作为一行，以此来表示二维矩阵
//	Index row = sourse.rows();
//	Index clo = sourse.cols();
//	in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)* row *  clo);
//	out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)* row *  clo);
//	for (int i = 0; i < row; i++)
//	{
//		for (int j = 0; j < clo; j++)
//		{
//			in[i*clo + j][0] = sourse(i, j);//通过col个数据表示一行，来逐行区分矩阵
//		}
//	}
//	p = fftw_plan_dft_2d(row, clo, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
//	fftw_execute(p);
//	fftw_destroy_plan(p);
//	for (int i = 0; i < row; i++)
//	{
//		for (int j = 0; j < clo; j++)
//		{
//			//result(i, j) = sqrt(pow(out[i*clo + j][0], 2) + pow(out[i*clo + j][1], 2));
//			result(i, j) = out[i*clo + j][0];
//		}
//	}
//	fftw_free(in); fftw_free(out);
//	return result;
//}
//MatrixXd  fftShit(MatrixXd sourse)
//{
//	int cx = sourse.cols() / 2;
//	int cy = sourse.rows() / 2;
//	MatrixXd q0 = sourse.block(0, 0, cx, cy);
//	MatrixXd q1 = sourse.block(cx, 0, cx, cy);
//	MatrixXd q2 = sourse.block(0, cy, cx, cy);
//	MatrixXd q3 = sourse.block(cx, cy, cx, cy);
//	MatrixXd temp(cx, cy);
//	temp = q0;
//	sourse.block(0, 0, cx, cy) = q3;
//	sourse.block(cx, cy, cx, cy) = q0;
//	temp = q1;
//	sourse.block(cx, 0, cx, cy) = q2;
//	sourse.block(0, cy, cx, cy) = q1;
//	return sourse;
//}
//MatrixXd  ifft(MatrixXd sourse)
//{
//	MatrixXd result = sourse;
//	//进行傅立叶变换  利用FFTW库
//	fftw_complex *in, *out;
//	fftw_plan p;
//	//申请一个行数乘以列数的一维向量，并且没列数个作为一行，以此来表示二维矩阵
//	Index row = sourse.rows();
//	Index col = sourse.cols();
//	in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)* row *  col);
//	out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)* row *  col);
//	for (int i = 0; i < row; i++)
//	{
//		for (int j = 0; j < col; j++)
//		{
//			in[i*col + j][0] = sourse(i, j);//通过col个数据表示一行，来逐行区分矩阵
//		}
//	}
//	p = fftw_plan_dft_2d(row, col, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
//	fftw_execute(p);
//	fftw_destroy_plan(p);
//	for (int i = 0; i < row; i++)
//	{
//		for (int j = 0; j < col; j++)
//		{
//			//result(i, j) = sqrt(pow(out[i*col + j][0], 2) + pow(out[i*col + j][1], 2));
//			//result(i, j) = result(i, j) / (row*col);
//			result(i, j) = out[i*col + j][0] / (row*col);
//		}
//	}
//	fftw_free(in); fftw_free(out);
//	return result;
//}
//Mat getMat(MatrixXd src)
//{
//	Mat aa = Mat::zeros(src.rows(), src.cols(), CV_32F);
//	aa = Mat_<double>(aa);
//	for (int i = 0; i < src.rows(); i++)
//	{
//		for (int j = 0; j < src.cols(); j++)
//		{
//			aa.at<double>(i, j) = src(i, j);
//		}
//	}
//	return aa;
//}
//void MyPrintf(string title, MatrixXd obj)
//{
//	cout << title << endl;
//	for (int i = 0; i < 10; i++)
//	{
//		for (int j = 0; j < 10; j++)
//		{
//			cout << obj(i, j) << "   ";
//		}
//		cout << endl;
//	}
//}
//
//int main()
//{
//	//1.获取彩色图像 
//	Mat srcImage;
//	srcImage = imread("img\\1.jpg");
//	if (!srcImage.data) { printf("Oh，no，读取srcImage错误~！\n"); return false; }
//	//分了色彩通道
//	vector<Mat>channels;
//	split(srcImage, channels);//分离色彩通道
//	Mat  matIb = channels.at(0);
//	Mat  matIg = channels.at(1);
//	Mat  matIr = channels.at(2);
//	MatrixXd Ib = getMatrix(matIb);
//	MatrixXd Ig = getMatrix(matIg);
//	MatrixXd Ir = getMatrix(matIr);
//	/********double********/
//	MatrixXd Ib_double = Ib.cast<double>();
//	MatrixXd Ig_double = Ig.cast<double>();
//	MatrixXd Ir_double = Ir.cast<double>();
//	//设定常量参数
//	int row = srcImage.rows;
//	int col = srcImage.cols;
//	int G = 192;	double b = -30;	int alpha = 125;	int beta = 46;
//	//	设定C的三个常量参数
//	int C_1 = 15; int C_2 = 80; int C_3 = 250;
//	//设定高斯函数模板 分别用三个常量参数计算高斯模板
//	MatrixXd gass_1 = GetGauss(C_1, row, col);
//	MatrixXd gass_2 = GetGauss(C_2, row, col);
//	MatrixXd gass_3 = GetGauss(C_3, row, col);
//	//1.转到对数域
//	MatrixXd Ib_log(row, col);
//	MatrixXd Ig_log(row, col);
//	MatrixXd Ir_log(row, col);
//	for (int i = 0; i < row; i++)
//	{
//		for (int j = 0; j < col; j++)
//		{
//			Ib_log(i, j) = log(Ib_double(i, j) + 1);
//			Ig_log(i, j) = log(Ig_double(i, j) + 1);
//			Ir_log(i, j) = log(Ir_double(i, j) + 1);
//		}
//	}
//
//	/******************R*******************/
//	//MyPrintf("Ir_double", Ir_double);
//	//R通道傅立叶变换
//	MatrixXd f_Ir = fft(Ir_double);
//
//
//
//	//高斯模板1傅立叶变换
//	MatrixXd fgauss = fft(gass_1);
//	//高斯模板1傅立叶变换后将频域中心移到零点
//	fgauss = fftShit(fgauss);
//	//卷积
//	MatrixXd con = f_Ir.cwiseProduct(fgauss);
//	//反变换
//	MatrixXd Rr = ifft(con);
//	//取最小值 
//	double min1 = Rr.minCoeff();
//
//	MatrixXd Rr_log(row, col);
//	for (int i = 0; i < row; i++)
//	{
//		for (int j = 0; j < col; j++)
//		{
//			Rr_log(i, j) = log(Rr(i, j) - min1 + 1);
//		}
//	}
//	MatrixXd Rr1 = Ir_log - Rr_log;
//
//	fgauss = fft(gass_2);
//	fgauss = fftShit(fgauss);
//	con = f_Ir.cwiseProduct(fgauss);
//	Rr = ifft(con);
//	min1 = Rr.minCoeff();
//	for (int i = 0; i < row; i++)
//	{
//		for (int j = 0; j < col; j++)
//		{
//			Rr_log(i, j) = log(Rr(i, j) - min1 + 1);
//		}
//	}
//	MatrixXd Rr2 = Ir_log - Rr_log;
//
//	fgauss = fft(gass_3);
//	fgauss = fftShit(fgauss);
//	con = f_Ir.cwiseProduct(fgauss);
//	Rr = ifft(con);
//	min1 = Rr.minCoeff();
//	for (int i = 0; i < row; i++)
//	{
//		for (int j = 0; j < col; j++)
//		{
//			Rr_log(i, j) = log(Rr(i, j) - min1 + 1);
//		}
//	}
//	MatrixXd Rr3 = Ir_log - Rr_log;
//
//	//%加权求和
//	Rr = 0.33*Rr1 + 0.34*Rr2 + 0.33*Rr3;
//	//计算CR
//	MatrixXd CRr(row, col);
//	for (int i = 0; i < row; i++)
//	{
//		for (int j = 0; j < col; j++)
//		{
//			CRr(i, j) = beta*(log(alpha*Ir_double(i, j) + 1) - log(Ir_double(i, j) + Ig_double(i, j) + Ib_double(i, j) + 1));
//		}
//	}
//	//%MSRCR
//	Rr = (G*CRr.cwiseProduct(Rr)).array() + b;
//	Mat tempIr = getMat(Rr);
//	normalize(tempIr, tempIr, 1.0, 0.0, NORM_MINMAX);
//
//
//	/******************G*******************/
//
//	/******************B*******************/
//
//
//
//	channels[0] = tempIr;
//	channels[1] = tempIg;
//	channels[2] = tempIb;
//	merge(channels, srcImage);
//	system("pause");
//	waitKey(0);
//}