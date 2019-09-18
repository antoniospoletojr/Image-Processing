#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <math.h>
#include <vector>
using namespace std;
using namespace cv;
#define SIZE 3
#define LEN floor(SIZE/2)

Mat filtro;

void filtroMedia(const Mat& inputImage, Mat& outputImage)
{
    int N = inputImage.rows;
    int M = inputImage.cols;
    float cont = 0;
    filtro = Mat::ones(SIZE,SIZE, CV_32FC1);
     for(int i=0; i<N; i++)
    {
        for(int j=0; j<M; j++)
        {
            cont = 0;
            for(int k=0; k<SIZE; k++)
            {
                for(int l=0; l<SIZE; l++)
                {
                     cont += inputImage.at<uchar>(i+k-LEN,j+l-LEN)*filtro.at<float>(k,l);
                }
            }
            outputImage.at<uchar>(i,j) = cont/9;
        }
    }
}

int main(int argc, char** argv )
{
    const char* fileName = "test.tif";
    Mat rawImage = imread(fileName, IMREAD_GRAYSCALE);
    Mat outputImage(rawImage.size(),rawImage.type());
    if (!rawImage.data)
    {
        printf("No image data \n");
        return -1;
    }
    imshow("input",rawImage);
    filtroMedia(rawImage,outputImage);
    imshow("output",outputImage);
    waitKey();
}