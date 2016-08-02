#include <iostream>
#include <opencv2\opencv.hpp>
#include <Eigen/Dense>
#pragma comment(lib, "D:\\LIB\\libfftw3-3.lib")
#pragma comment(lib, "D:\\LIB\\libfftw3f-3.lib")
#pragma comment(lib, "D:\\LIB\\libfftw3l-3.lib")
#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>

using namespace cv;
using namespace std;
using namespace Eigen;

//-----------------------------
//����������Mat�ĵ�ͨ����ת��ΪMatrix��float����
//-----------------------------
MatrixXd getMatrix(Mat source)
{
	MatrixXd result(source.rows, source.cols);
	uchar* inputImageData1 = source.data;
	for (int i = 0; i < source.rows; i++)
	{
		for (int j = 0; j < source.cols; j++)
		{
			int index = i*source.cols + j;
			double b = inputImageData1[index];
			result(i, j) = b;
		}
	}
	return result;
}
MatrixXd GetGauss(int _sigma, int rows, int cols)
{
	MatrixXd result(rows, cols);
	double sum = 0.0000000;
	//��x����ֵ
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			double bx = -(double)((cols - 1.00) / 2.00) + (double)j;
			double by = -(double)((rows - 1.00) / 2.00) + (double)i;
			double num = -(bx*bx + by*by) / (2 * _sigma*_sigma);
			result(i, j) = exp(num);
			sum += result(i, j);
		}
	}
	result = result / (sum + 0.001);
	return result;
}
Mat fft2(Mat src)
{
	Mat Fourier = Mat::zeros(src.rows, src.cols, CV_32F);
	Mat planes[] = { Mat_<float>(src), Mat::zeros(src.size(), CV_32F) }; //����һ��planes����
	merge(planes, 2, Fourier);
	dft(Fourier, Fourier, cv::DftFlags::DFT_COMPLEX_OUTPUT);

	/*vector<Mat> vec;
	split(Fourier, vec);
	return vec.at(0);*/
	return Fourier;
}
MatrixXd  fft(MatrixXd sourse)
{
	MatrixXd result = sourse;
	//���и���Ҷ�任  ����FFTW��
	fftw_complex *in, *out;
	fftw_plan p;
	//����һ����������������һά����������û��������Ϊһ�У��Դ�����ʾ��ά����
	Index row = sourse.rows();
	Index clo = sourse.cols();
	in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)* row *  clo);
	out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)* row *  clo);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < clo; j++)
		{
			in[i*clo + j][0] = sourse(i, j);//ͨ��col�����ݱ�ʾһ�У����������־���
		}
	}
	p = fftw_plan_dft_2d(row, clo, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < clo; j++)
		{
			result(i, j) = sqrt(pow(out[i*clo + j][0], 2) + pow(out[i*clo + j][1], 2));
			//result(i, j) = out[i*clo + j][0];
		}
	}
	fftw_free(in); fftw_free(out);
	return result;
}
MatrixXd  fftShit(MatrixXd sourse)
{
	int cx = sourse.rows() / 2;
	int cy = sourse.cols() / 2;
	MatrixXd q0 = sourse.block(0, 0, cx, cy);
	MatrixXd q1 = sourse.block(cx, 0, cx, cy);
	MatrixXd q2 = sourse.block(0, cy, cx, cy);
	MatrixXd q3 = sourse.block(cx, cy, cx, cy);
	MatrixXd temp(cx, cy);
	temp = q0;
	sourse.block(0, 0, cx, cy) = q3;
	sourse.block(cx, cy, cx, cy) = q0;
	temp = q1;
	sourse.block(cx, 0, cx, cy) = q2;
	sourse.block(0, cy, cx, cy) = q1;
	return sourse;
}
MatrixXd  ifft(MatrixXd sourse)
{
	MatrixXd result = sourse;
	//���и���Ҷ�任  ����FFTW��
	fftw_complex *in, *out;
	fftw_plan p;
	//����һ����������������һά����������û��������Ϊһ�У��Դ�����ʾ��ά����
	Index row = sourse.rows();
	Index col = sourse.cols();
	in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)* row *  col);
	out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)* row *  col);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			in[i*col + j][0] = sourse(i, j);//ͨ��col�����ݱ�ʾһ�У����������־���
		}
	}
	p = fftw_plan_dft_2d(row, col, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			//result(i, j) = sqrt(pow(out[i*col + j][0], 2) + pow(out[i*col + j][1], 2));
			//result(i, j) = result(i, j) / (row*col);
			result(i, j) = out[i*col + j][0] / (row*col);
		}
	}
	fftw_free(in); fftw_free(out);
	return result;
}
Mat getMat(MatrixXd src)
{
	Mat aa = Mat::zeros(src.rows(), src.cols(), CV_32F);
	aa = Mat_<double>(aa);
	for (int i = 0; i < src.rows(); i++)
	{
		for (int j = 0; j < src.cols(); j++)
		{
			aa.at<double>(i, j) = src(i, j);
		}
	}
	return aa;
}
void MyPrintf(string title, MatrixXd obj)
{
	cout << title << endl;
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			cout << obj(i, j) << "   ";
		}
		cout << endl;
	}
}
Mat dealSingleChannel(int flag, MatrixXd Ir_double, MatrixXd Ig_double, MatrixXd Ib_double, MatrixXd r_log, MatrixXd g_log, MatrixXd b_log, MatrixXd gass_1, MatrixXd gass_2, MatrixXd gass_3)
{

	Index row = Ir_double.rows();
	Index col = Ir_double.cols();
	//�Ե�ǰͨ�����и���Ҷ�任
	MatrixXd f_Ichannel(row, col);
	MatrixXd fgauss(row, col);//��Ÿ�˹ģ�帵��Ҷ�任��Ľ��
	MatrixXd con(row, col);//��ž���Ľ��
	MatrixXd Rchannel(row, col);//��ž������Ҷ���任��Ľ��
	MatrixXd R_log(row, col);//��ŵ�Ƶͼ��
	MatrixXd Rchannel1(row, col);//��Ÿ�Ƶͼ��
	MatrixXd Rchannel2(row, col);//��Ÿ�Ƶͼ��
	MatrixXd Rchannel3(row, col);//��Ÿ�Ƶͼ��

	//�Ե�ǰͨ�����и���Ҷ�任
	switch (flag)
	{
	case 0://R
		f_Ichannel = fft(Ir_double);
		break;
	case 1://G
		f_Ichannel = fft(Ig_double);
		break;
	case 2://B
		f_Ichannel = fft(Ib_double);
		break;
	}
	double min1 = 0;
	/***********��˹ģ�� 1************/
	//��˹ģ�帵��Ҷ�任
	fgauss = fft(gass_1);
	//��Ƶ�������Ƶ����
	fgauss = fftShit(fgauss);
	//���
	con = f_Ichannel.cwiseProduct(fgauss);
	//����Ҷ���任
	Rchannel = ifft(con);
	//ȡ��Сֵ �õ���Ƶͼ��
	min1 = Rchannel.minCoeff();

	R_log = (Rchannel.array() - min1 + 1).log();
	//��ԭͼ��ȥ��Ƶͼ��õ���Ƶ��ǿͼ��
	switch (flag)
	{
	case 0://R
		Rchannel1 = r_log - R_log;
		break;
	case 1://G
		Rchannel1 = g_log - R_log;
		break;
	case 2://B
		Rchannel1 = b_log - R_log;
		break;
	}



	/***********��˹ģ�� 2************/
	//��˹ģ��1����Ҷ�任
	fgauss = fft(gass_2);
	//��Ƶ�������Ƶ����
	fgauss = fftShit(fgauss);
	//���
	con = f_Ichannel.cwiseProduct(fgauss);
	//����Ҷ���任
	Rchannel = ifft(con);
	//ȡ��Сֵ �õ���Ƶͼ��
	min1 = Rchannel.minCoeff();
	R_log = (Rchannel.array() - min1 + 1).log();
	//��ԭͼ��ȥ��Ƶͼ��õ���Ƶ��ǿͼ��
	switch (flag)
	{
	case 0://R
		Rchannel2 = r_log - R_log;
		break;
	case 1://G
		Rchannel2 = g_log - R_log;
		break;
	case 2://B
		Rchannel2 = b_log - R_log;
		break;
	}


	/***********��˹ģ�� 3************/
	//��˹ģ��1����Ҷ�任
	fgauss = fft(gass_3);
	//��Ƶ�������Ƶ����
	fgauss = fftShit(fgauss);
	//���
	con = f_Ichannel.cwiseProduct(fgauss);
	//����Ҷ���任
	Rchannel = ifft(con);
	//ȡ��Сֵ �õ���Ƶͼ��
	min1 = Rchannel.minCoeff();
	R_log = (Rchannel.array() - min1 + 1).log();
	//��ԭͼ��ȥ��Ƶͼ��õ���Ƶ��ǿͼ��
	switch (flag)
	{
	case 0://R
		Rchannel3 = r_log - R_log;
		break;
	case 1://G
		Rchannel3 = g_log - R_log;
		break;
	case 2://B
		Rchannel3 = b_log - R_log;
		break;
	}

	//�Ը�Ƶ��ǿͼ����м�Ȩ
	Rchannel = 0.33*Rchannel1 + 0.34*Rchannel2 + 0.33*Rchannel3;
	//�����ɫ����
	int G = 192;	double b = -30;	int alpha = 125;	int beta = 46;
	MatrixXd CRr(row, col);
	switch (flag)
	{
		case 0://R
			CRr = beta * ((Ir_double.array()*alpha + 1).log() - (Ir_double.array() + Ig_double.array() + Ib_double.array() + 1).log()).array();
			break;
		case 1://G
			CRr = beta * ((Ig_double.array()*alpha + 1).log() - (Ir_double.array() + Ig_double.array() + Ib_double.array() + 1).log()).array();
			break;
		case 2://B
			CRr = beta * ((Ib_double.array()*alpha + 1).log() - (Ir_double.array() + Ig_double.array() + Ib_double.array() + 1).log()).array();
			break;
	}
	//���ù�ʽ R=G*R+O��ͼ����н�һ������
	Rchannel = (G*CRr.cwiseProduct(Rchannel)).array() + b;
	//ͼ���һ��
	Mat result = getMat(Rchannel);
	normalize(result, result, 1.0, 0.0, NORM_MINMAX);
	return result;
}
int main()
{
	//1.��ȡ��ɫͼ�� 
	Mat srcImage;
	srcImage = imread("img\\13.jpg");
	if (!srcImage.data) { printf("Oh��no����ȡsrcImage����~��\n"); return false; }
	//����ɫ��ͨ��
	vector<Mat>channels;
	split(srcImage, channels);//����ɫ��ͨ��
	Mat  matIb = channels.at(0);
	Mat  matIg = channels.at(1);
	Mat  matIr = channels.at(2);
	MatrixXd Ib = getMatrix(matIb);
	MatrixXd Ig = getMatrix(matIg);
	MatrixXd Ir = getMatrix(matIr);
	/********double********/
	MatrixXd Ib_double = Ib.cast<double>();
	MatrixXd Ig_double = Ig.cast<double>();
	MatrixXd Ir_double = Ir.cast<double>();
	//�趨��������
	int row = srcImage.rows;
	int col = srcImage.cols;
	//	�趨C��������������
	int C_1 = 15; int C_2 = 80; int C_3 = 250;
	//�趨��˹����ģ�� �ֱ��������������������˹ģ��
	MatrixXd gass_1 = GetGauss(C_1, row, col);
	MatrixXd gass_2 = GetGauss(C_2, row, col);
	MatrixXd gass_3 = GetGauss(C_3, row, col);
	//1.ת��������
	MatrixXd Ib_log = (Ib_double.array() + 1).log();
	MatrixXd Ig_log = (Ig_double.array() + 1).log();
	MatrixXd Ir_log = (Ir_double.array() + 1).log();


	/******************R*******************/
	Mat tempIr = dealSingleChannel(0, Ir_double, Ig_double, Ib_double, Ir_log, Ig_log, Ib_log, gass_1, gass_2, gass_3);
	/******************G*******************/
	Mat tempIg = dealSingleChannel(1, Ir_double, Ig_double, Ib_double, Ir_log, Ig_log, Ib_log, gass_1, gass_2, gass_3);
	/******************B*******************/
	Mat tempIb = dealSingleChannel(2, Ir_double, Ig_double, Ib_double, Ir_log, Ig_log, Ib_log, gass_1, gass_2, gass_3);
	channels[0] = tempIb;

	channels[1] = tempIg;
	channels[2] = tempIr;
	Mat result = srcImage;
	merge(channels, result);
	system("pause");
	waitKey(0);
}