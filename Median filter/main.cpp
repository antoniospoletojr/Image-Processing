#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <math.h>
#include <vector>
#include <algorithm>
using namespace std;
using namespace cv;
#define SIZE 3
#define LEN floor(SIZE/2)

void filtroMedia(const Mat& inputImage, Mat& outputImage)
{
    int N = inputImage.rows;
    int M = inputImage.cols;
    vector<int> lista(SIZE*SIZE,0);
     for(int i=0; i<N; i++)
    {
        for(int j=0; j<M; j++)
        {
            for(int k=0; k<SIZE; k++)
            {
                for(int l=0; l<SIZE; l++)
                {
                     lista.at(k*SIZE+l) = inputImage.at<uchar>(i+k-LEN,j+l-LEN);
                }
            }
            sort(lista.begin(),lista.end());
            outputImage.at<uchar>(i,j) = lista.at((SIZE*SIZE)/2);
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