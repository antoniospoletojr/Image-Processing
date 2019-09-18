#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#define PAD floor(3/2)

Mat kernel4 = (Mat_<char>(3,3) << 0, 1, 0,
                                  1, 1, 1,
                                  0, 1, 0);
float accum=0;

void frameMatrix(const Mat& rawImage, Mat& outputImage)
{
    int N = rawImage.rows;
    int M = rawImage.cols;
    outputImage = Mat(N+2*PAD,M+2*PAD, rawImage.type(),Scalar(255));  
    for(int i=PAD; i<N; i++)
    {
        for(int j=PAD; j<M; j++)
        {
            outputImage.at<uchar>(i,j) = rawImage.at<uchar>(i-PAD,j-PAD);
        }
    }
    //imshow("framed image", outputImage);
}

void erosione(Mat& inputImage)
{
    int N = inputImage.rows;
    int M = inputImage.cols;
    Mat outputImage = inputImage.clone();
    int sum;
    for(int i=PAD; i<N-PAD; i++)
    {
        for(int j=PAD; j<M-PAD; j++)
        {  
            sum=0;
            for(int k=-PAD; k<=PAD; k++)
            {
                for(int l=-PAD; l<=PAD; l++)
                {
                    sum+=inputImage.at<uchar>(i+k,j+l)*kernel4.at<char>(k+PAD,l+PAD);
                }
            }
            if(sum<255*5)
            {
                if(inputImage.at<uchar>(i,j)>0)
                {
                    outputImage.at<uchar>(i,j) = 0;
                    accum++;
                }
            }
        }
    }
    inputImage=outputImage.clone();
}

void dilatazione(Mat& inputImage)
{
    int N = inputImage.rows;
    int M = inputImage.cols;
    Mat outputImage = inputImage.clone();
    int sum;
    for(int i=PAD; i<N-PAD; i++)
    {
        for(int j=PAD; j<M-PAD; j++)
        {  
            sum=0;
            for(int k=-PAD; k<=PAD; k++)
            {
                for(int l=-PAD; l<=PAD; l++)
                {
                    sum+=inputImage.at<uchar>(i+k,j+l)*kernel4.at<char>(k+PAD,l+PAD);
                }
            }
            if(sum==0)
                outputImage.at<uchar>(i,j) = inputImage.at<uchar>(i,j);
            else
                outputImage.at<uchar>(i,j)= 255;
        }
    }
    inputImage=outputImage.clone();
}

void sogliatura(Mat& inputImage)
{
    int N = inputImage.rows;
    int M = inputImage.cols;
    int min = 255, max=0, mean;
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<M; j++)
        {
            if(inputImage.at<uchar>(i,j)>max)
                max=inputImage.at<uchar>(i,j);
            if(inputImage.at<uchar>(i,j)<min)
                min=inputImage.at<uchar>(i,j);
        }
    }
    mean=(int)(max+min)/2;
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<M; j++)
        {
            if(inputImage.at<uchar>(i,j)>mean)
                inputImage.at<uchar>(i,j)=255;
            else
                inputImage.at<uchar>(i,j)=0;
        }
    }
    //imshow("immagine sogliata", inputImage);
}

void chiusura(Mat& inputImage)
{
    dilatazione(inputImage);
    erosione(inputImage);
}

void apertura(Mat& inputImage)
{
    erosione(inputImage);
    dilatazione(inputImage);
}

void trasformataDistanza(Mat& inputImage)
{
    int N = inputImage.rows;
    int M = inputImage.cols;
    Mat dt = Mat::zeros(inputImage.size(), CV_32FC1);
    accum = (N*M)-sum(inputImage)[0]/255;
    for(int i=PAD; i<N-PAD; i++)
    {
        for(int j=PAD; j<M-PAD; j++)
        {
            dt.at<float>(i,j) += inputImage.at<uchar>(i,j);
        }
    }
    while(accum < N*M)
    {
        erosione(inputImage);
        for(int i=PAD; i<N-PAD; i++)
        {
            for(int j=PAD; j<M-PAD; j++)
            {
                dt.at<float>(i,j) += inputImage.at<uchar>(i,j);
            }
        }
    }
    normalize(dt,inputImage,0,255, NORM_MINMAX, CV_8UC1);
}

void trasformataDistanza2(Mat& inputImage)
{
    int N = inputImage.rows;
    int M = inputImage.cols;
    int min=INT_MAX;
    Mat dt = Mat::zeros(inputImage.size(), CV_32FC1);
    for(int i=PAD; i<N-PAD; i++)
    {
        for(int j=PAD; j<M-PAD; j++)
        {
            min = inputImage.at<uchar>(i-1,j);          //NORTH
            if(inputImage.at<uchar>(i-1,j-1) < min)     //NORTH-WEST
                min = inputImage.at<uchar>(i-1,j-1);
            if(inputImage.at<uchar>(i,j-1) < min)       //WEST
                min = inputImage.at<uchar>(i,j-1);
            if(inputImage.at<uchar>(i-1,j+1) < min)     //NORTH-EAST
                min = inputImage.at<uchar>(i-1,j+1);
            dt.at<float>(i,j) = min+1;
        }
    }

    for(int i=N-PAD-1; i>=PAD; i--)
    {
        for(int j=M-PAD-1; j>=PAD; j--)
        {
            min = dt.at<float>(i+1,j);          //SOUTH
            if(dt.at<float>(i+1,j+1) < min)     //SOUTH-EAST
                min = dt.at<float>(i+1,j+1);
            if(dt.at<float>(i,j+1) < min)       //EAST
                min = dt.at<float>(i,j+1);
            if(dt.at<float>(i+1,j-1) < min)     //SOUTH-WEST
                min = dt.at<float>(i+1,j-1);
            if(dt.at<float>(i,j)>min)
                dt.at<float>(i,j) = min+1;
        }
    }
    imshow("sd", dt);
    normalize(dt,inputImage,0,255, NORM_MINMAX, CV_8UC1);
}

int main(int argc, char** argv )
{
    const char* fileName = "topolino.jpg";
    Mat rawImage = imread(fileName, IMREAD_GRAYSCALE);                          //immagine presa in input
    Mat paddedImage;
    if (!rawImage.data)
    {
        printf("No image data \n");
        return -1;
    }
    frameMatrix(rawImage,paddedImage);
    sogliatura(paddedImage);
    threshold(paddedImage,paddedImage,0,255, THRESH_BINARY_INV); //inversione
    //Canny(paddedImage,paddedImage,100,150,3);
  //  imshow("immagine", paddedImage);
    trasformataDistanza(paddedImage);
    imshow("immagine dt", paddedImage);
    waitKey(0);
    return 0;
}

    // Mat test = paddedImage.clone();
    // Mat kernel = getStructuringElement(MORPH_CROSS, Size(3,3), Point(-1,1));
    // dilate(paddedImage, test, kernel, Point(-1,-1), 1);
    // morphologyEx(paddedImage, test, MORPH_CLOSE, kernel, Point(-1,-1), 1);
    // imshow("immagine chiusa opencv", test);