#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#define GAUSS_LEN 5 //DIMENSIONI KERNEL PER SMOOTHING GAUSSIANO
#define GAUSS_PAD floor(GAUSS_LEN/2)
#define PAD floor(5/2)

Mat horizSobel = (Mat_<char>(3,3) <<-1, -2, -1,
                                    0,  0,  0,
                                    1,  2,  1);
Mat vertSobel = (Mat_<char>(3,3) <<-1, 0, 1,
                                    -2, 0, 2,
                                    -1, 0, 1);

class Triad
{
    public:
        int x;
        int y;
        float lambda;

    Triad(int x, int y, float lambda)
    {
        this->x=x;
        this->y=y;
        this->lambda=lambda;
    }
};

void displayHistogram(const Mat& image)
{
    int N = image.rows;
    int M = image.cols;
    vector<int> histogram(256,0);
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            histogram.at(image.at<uchar>(i,j))++;
    //FIND THE MAX VALUE OF THE BINS
    int max = histogram.at(0);
    for(int i = 0; i < 256; i++)
        if(max < histogram.at(i))
            max = histogram.at(i);
    int hist_width = 1024;
    int hist_height = 400;
    int bin_widht = round((float)hist_width/256);
    Mat histogramImage(hist_height+24, hist_width+48, CV_8UC1, Scalar(255, 255, 255));
    for(int i=0; i<256; i++)
        histogram.at(i) = ((double)histogram.at(i)/max)*hist_height;
    //DRAW THE LINE
    for(int i = 0; i < 256; i++)
      //line(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
        line(histogramImage, Point(bin_widht*(i)+24, hist_height), Point(bin_widht*(i)+24, hist_height - histogram.at(i)),Scalar(0,0,0), 2, 8, 0);
    imshow("histogram", histogramImage);
    
}

void sobel(const Mat& inputImage, Mat& gx, Mat& gy)
{
    int N = inputImage.rows;
    int M = inputImage.cols;
    int xGradient, yGradient;
    for(int i=PAD; i<N-PAD; i++)
    {
        for(int j=PAD; j<M-PAD; j++)
        {
            xGradient = 0, yGradient = 0;
            for(int k=0; k<3; k++)
            {
                for(int l=0; l<3; l++)
                {
                    xGradient += inputImage.at<uchar>(i+k-1,j+l-1)*horizSobel.at<char>(k,l);
                    yGradient += inputImage.at<uchar>(i+k-1,j+l-1)*vertSobel.at<char>(k,l);
                }
            }
            gx.at<uchar>(i,j) = abs(xGradient);;
            gy.at<uchar>(i,j) = abs(yGradient);;
        }
    }
    // imshow("gx", gx);
    // imshow("gy", gy);
}

void gaussianFilter(const Mat& inputImage, Mat& filterImage)
{
	Mat kernel = Mat::zeros(GAUSS_LEN,GAUSS_LEN,CV_32FC1);
	Mat temp = Mat(inputImage.size(),inputImage.type());
	int N = inputImage.rows;
 	int M = inputImage.cols;
 	short var=4;
 	float min,count,sum=0;
 	for(int i=-GAUSS_PAD;i<=GAUSS_PAD;i++)
 		for(int j=-GAUSS_PAD;j<=GAUSS_PAD;j++)
			kernel.at<float>(i+GAUSS_PAD,j+GAUSS_PAD) = exp(-((pow(i,2)/(2*var))+(pow(j,2)/(2*var))));			
    min=kernel.at<float>(0,0);
    float B=1/min;
 	for(int i=-GAUSS_PAD;i<=GAUSS_PAD;i++)
 		for(int j=-GAUSS_PAD;j<=GAUSS_PAD;j++)
			kernel.at<float>(i+GAUSS_PAD,j+GAUSS_PAD) = round(kernel.at<float>(i+GAUSS_PAD,j+GAUSS_PAD)*B);
	for(int i=-GAUSS_PAD;i<=GAUSS_PAD;i++)
    {
        // cout << endl;
     	for(int j=-GAUSS_PAD;j<=GAUSS_PAD;j++)
         {
			sum+=kernel.at<float>(i+GAUSS_PAD,j+GAUSS_PAD);
            // cout << kernel.at<float>(i+GAUSS_PAD,j+GAUSS_PAD) << " ";
         }
    }
    // cout << endl;
    for(int i=PAD;i<N-PAD;i++)
	{
		for(int j=PAD;j<M-PAD;j++)
		{
			count=0;
			for(int k=0;k<kernel.rows;k++)
			{	
				for(int l=0;l<kernel.cols;l++)
					{
						count += inputImage.at<uchar>(i+k-GAUSS_PAD,j+l-GAUSS_PAD)*kernel.at<float>(k,l);
				    }	
			}
			temp.at<uchar>(i,j) = round(count/sum);
		}
	}
	filterImage=temp.clone();
}

void frameMatrix(const Mat& rawImage, Mat& outputImage)
{
    int N = rawImage.rows;
    int M = rawImage.cols;
    outputImage = Mat::zeros(N+2*PAD,M+2*PAD, rawImage.type());  
    for(int i=PAD; i<N; i++)
    {
        for(int j=PAD; j<M; j++)
        {
            outputImage.at<uchar>(i,j) = rawImage.at<uchar>(i-PAD,j-PAD);
        }
    }
    //imshow("framed image", image);
}

void suppressCorners(vector<Triad>& list)
{
	sort(list.begin(), list.end(), [](Triad& l, Triad& r) { return l.lambda > r.lambda;});
    short offset = 4;
    float radius = sqrt(pow(offset,2)+pow(offset,2)), distance;
    for(int i=0; i<list.size(); i++)
    {
        for(int j=i+1; j<list.size(); j++)
        {
            distance = sqrt(pow(list.at(i).x-list.at(j).x,2)+pow(list.at(i).y-list.at(j).y,2));
            if(distance<=radius)
            {
                list.erase(list.begin()+j);
                j--;
            }       
        }
    }
}


float getEigenValue(const Mat& gx, const Mat& gy, short windowSize, int x, int y)
{
    Mat eigenVal = Mat::zeros(2, 1, CV_32FC1);
    Mat covarianceMatrix = Mat::zeros(2,2,CV_32FC1);
    int len = floor(windowSize/2);
    float lambda;
    for(int i=-len; i<=len; i++)
    {
        for(int j=-len; j<=len; j++)
        {
            covarianceMatrix.at<float>(0,0) += (float)pow(gx.at<uchar>(x+i,y+j),2);
            covarianceMatrix.at<float>(0,1) += (float)gx.at<uchar>(x+i,y+j)*gy.at<uchar>(x+i,y+j);
            covarianceMatrix.at<float>(1,0) += (float)gx.at<uchar>(x+i,y+j)*gy.at<uchar>(x+i,y+j);
            covarianceMatrix.at<float>(1,1) += (float)pow(gy.at<uchar>(x+i,y+j),2);
        }
    }
    eigen(covarianceMatrix,eigenVal);
    if(eigenVal.at<float>(0,0)>eigenVal.at<float>(1,0))
        lambda = eigenVal.at<float>(1,0);
    else
        lambda = eigenVal.at<float>(0,0);
    if(lambda > 6000)
        return lambda;
    return 0;
}

void harris(Mat& image, const Mat& gx, const Mat& gy)
{
    vector<Triad> list;
    int N = gx.rows;
    int M = gy.cols;  
    float eigenValue;
    for(int i=PAD; i<N-PAD; i++)
    {
        for(int j=PAD; j<M-PAD; j++)
        {
            eigenValue=getEigenValue(gx,gy,3,i,j);
            if(eigenValue!=0) //è un corner
                list.push_back(Triad(i,j,eigenValue));
        }
    }
    suppressCorners(list);
    
    for(int i=0; i<list.size(); i++)
    {
        if(list.at(i).lambda>0)
            circle(image,Point(list.at(i).y,list.at(i).x),5, Scalar(100), 1, 4);
    }
    imshow("corners", image);
}

int main(int argc, char** argv )
{
    const char* fileName = "Immagine_Harris.jpg";
    Mat rawImage = imread(fileName, IMREAD_GRAYSCALE);                          //immagine presa in input
    Mat paddedImage;
    if (!rawImage.data)
    {
        printf("No image data \n");
        return -1;
    }
    GaussianBlur(rawImage, rawImage, Size(5,5), 3, 3, 4 ); // threshold(rawImage,rawImage, 170, 255, THRESH_BINARY);
    threshold(rawImage,rawImage, 170, 255, THRESH_BINARY);
    morphologyEx(rawImage,rawImage, MORPH_OPEN, Mat());
    frameMatrix(rawImage,paddedImage);
    Mat gx(paddedImage.size(), CV_8UC1);
    Mat gy(paddedImage.size(), CV_8UC1);
    sobel(paddedImage,gx,gy);
    gaussianFilter(gx,gx);
    gaussianFilter(gy,gy);
    harris(paddedImage,gx,gy);
    
    waitKey(0);
    return 0;
}