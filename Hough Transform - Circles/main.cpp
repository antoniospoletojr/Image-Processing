#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void displayAccumulator(Mat& accumulator, int radius)
{
    Point point;
    double max;
    minMaxLoc(accumulator,NULL,&max,NULL,&point);
    cout << "Raggio: " << radius << ". Voto massimo: " << max << endl;
    Mat parameters(accumulator.size(), CV_8UC1);
    normalize(accumulator, parameters, 0, 255, NORM_MINMAX);
    cvtColor(parameters, parameters, COLOR_GRAY2BGR);
    String name = to_string(radius);
    name = name + ".png";
    imwrite(name.c_str(), parameters);
}

void houghTransform(Mat& inputImage, Mat& rawImage, int minRadius, int maxRadius, int threshold)
{
    int N = inputImage.rows;
    int M = inputImage.cols;
    int xc,yc;
    cvtColor(rawImage, rawImage, COLOR_GRAY2BGR);
    for(int r=minRadius; r<maxRadius; r++)
    {
        Mat accumulator = Mat::zeros(N, M, CV_32FC1);
        for(int y=0; y<N; y++)//verso il basso = Y
        {
            for(int x=0; x<M; x++)//verso destra = X
            {
                if(inputImage.at<uchar>(y,x)!=0)
                {
                    for(int theta=0; theta<360; theta++)
                    {
                        xc = x-r*cos(theta*M_PI/180);
                        yc = y-r*sin(theta*M_PI/180);
                        if(xc>=0 && yc>=0 && xc<M && yc<N)
                            accumulator.at<float>(yc,xc)++;
                    }
                }
            }
        }
        displayAccumulator(accumulator,r); //stampa il centro più votato per quel raggio, in modo da stimare un threshold ottimale e scrive l'immagine r-ima
        for(int i=0; i<N; i++)
        {
            for(int j=0; j<M; j++)
            {
                if(accumulator.at<float>(i,j)>threshold)
                    circle(rawImage, Point(j,i), r, Scalar(0,0,255), 2, 8);
            }
        }
    }
    imwrite("result.jpg", rawImage);
    imshow("cerchi", rawImage);
}

int main(int argc, char** argv )
{
    const char* fileName = "eye.png";
    Mat rawImage = imread(fileName, IMREAD_GRAYSCALE);//immagine presa in input
    resize(rawImage,rawImage, Size(rawImage.cols/2,rawImage.rows/2));
    Mat edgesImage(rawImage.size(), CV_8UC1);
    Mat blurImage = rawImage.clone();
    if (!rawImage.data)
    {
        printf("No image data \n");
        return -1;
    }
    blur(rawImage,blurImage, Size(5,5), Point(-1,-1), 4);
    Canny(blurImage,edgesImage,70,90,3);
    imshow("canny", edgesImage);
    //imshow("canny", edgesImage);
    houghTransform(edgesImage,rawImage,43,44,120);
    cvtColor(rawImage,rawImage, COLOR_BGR2GRAY);
    houghTransform(edgesImage,rawImage,114,115,100);
    waitKey(0);
    return 0;
}
