#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;


int g_d = 15;
int g_sigmaColor = 30;
int g_sigmaSpace = 10;

Mat image1;
Mat image2;

void on_Trackbar(int, void*)
{
    bilateralFilter(image1, image2, g_d, g_sigmaColor, g_sigmaSpace);
    imshow("output", image2);
}


int main(int argc, char **argv)
{
	string bfInNa;
	cout<<"请输入本路径下的图片名称，spaceSigma，colorSigma（e.g. meinv.jpg 10 30）："<<endl;
	cin>>bfInNa>>g_sigmaSpace>>g_sigmaColor;
	string bfin = "./"+bfInNa;

    Mat image1 = imread(bfin);
    if (image1.empty())
    {
        cout << "Could not load image ... " << endl;
        return  -1;
    }

    image2 = Mat::zeros(image1.rows, image1.cols, image1.type());
    bilateralFilter(image1, image2, g_d, g_sigmaColor, g_sigmaSpace);

    namedWindow("output");

    createTrackbar("核直径","output", &g_d, 50, on_Trackbar);
    createTrackbar("颜色空间方差","output", &g_sigmaColor, 100, on_Trackbar);
    createTrackbar("坐标空间方差","output", &g_sigmaSpace, 100, on_Trackbar);

    imshow("input", image1);
    imshow("output", image2);

    string bfout = "./s"+to_string((int)g_sigmaSpace)+'c'+to_string((int)g_sigmaColor)+"-opencv-"+bfInNa; 
	cv::imwrite(bfout, image2);
    //cv::imwrite("/home/eli/gitDir/bf/ocout.bmp", image2);

    waitKey(0);
    return 0;
}