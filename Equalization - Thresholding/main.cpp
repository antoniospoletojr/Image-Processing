#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <math.h>
#include <vector>
using namespace std;
using namespace cv;


void equalizzazione(const Mat& rawImage, Mat& outputImage)
{
    int L = 256;
    int N = rawImage.rows;
    int M = rawImage.cols;
    vector<int> istogramma(L,0), istogrammaEqualizzato(L,0);
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<M; j++)
        {
            istogramma.at(rawImage.at<uchar>(i,j))++;
        }
    }

    for(int i=0; i<L; i++)
    {
        int accumulatore = 0;
        for(int j=0; j<i; j++)
        {
            accumulatore += istogramma.at(j);
        }
        istogrammaEqualizzato.at(i) = ((float)(L-1)/(N*M))*accumulatore;
    }

    for(int i=0; i<N; i++)
    {
        for(int j=0; j<M; j++)
        {
            outputImage.at<uchar>(i,j) = istogrammaEqualizzato.at(rawImage.at<uchar>(i,j));
        }
    }
}

void sogliatura(const Mat& rawImage, Mat& outputImage, int threshold)
{
    int N = rawImage.rows;
    int M = rawImage.cols;
    short max = 0, min = 255;
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<M; j++)
        {
            if(rawImage.at<uchar>(i,j) > max)
                max=rawImage.at<uchar>(i,j);
            if(rawImage.at<uchar>(i,j) < min)
                min=rawImage.at<uchar>(i,j);
        }
    }
    threshold = (max+min)/2;
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<M; j++)
        {
            if(rawImage.at<uchar>(i,j) >= threshold)
                outputImage.at<uchar>(i,j) = 255;
            else
                outputImage.at<uchar>(i,j) = 0;
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
    equalizzazione(rawImage,outputImage);
    sogliatura(outputImage,outputImage,55);
    imshow("output",outputImage);
    waitKey();
}