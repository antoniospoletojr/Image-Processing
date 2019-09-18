#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;



// void sobel(const Mat& inputImage, Mat& gx, Mat& gy, Mat& alfa)
// {
//     int N = inputImage.rows;
//     int M = inputImage.cols;
//     int xGradient, yGradient;
//     for(int i=PAD; i<N-PAD; i++)
//     {
//         for(int j=PAD; j<M-PAD; j++)
//         {
//             xGradient = 0, yGradient = 0;
//             for(int k=0; k<3; k++)
//             {
//                 for(int l=0; l<3; l++)
//                 {
//                     xGradient += inputImage.at<uchar>(i+k-1,j+l-1)*horizSobel.at<char>(k,l);
//                     yGradient += inputImage.at<uchar>(i+k-1,j+l-1)*vertSobel.at<char>(k,l);
//                 }
//             }
//             xGradient = abs(xGradient);
//             yGradient = abs(yGradient);
//             gx.at<uchar>(i,j) = xGradient > 255? 255: xGradient;
//             gy.at<uchar>(i,j) = yGradient > 255? 255: yGradient;
//             alfa.at<float>(i,j) = (atan2(yGradient,xGradient)*180)/M_PI;
//             magnitudo.at<float>(i,j) = xGradient + yGradient;
//         }
//     }
//     //imshow("gx", gx);
//     //imshow("gy", gy);
//     //imshow("magnitudo", magnitudo);
//     //imshow("alfa", alfa);
// }

void drawLines(Mat& rawImage, vector<Point2d> rette)
{
    cvtColor(rawImage, rawImage, COLOR_GRAY2BGR);
    double rho, theta, x0, y0;
    Point a,b;
    for(int i=0; i<rette.size(); i++)
    {
        rho = rette.at(i).y;
        theta = rette.at(i).x;
        x0 = round(rho * cos(theta));
        y0 = round(rho * sin(theta));
        a.x = round(x0 - 1000 * (sin(theta)));
        a.y = round(y0 + 1000 * (cos(theta)));
        b.x = round(x0 + 1000 * (sin(theta)));
        b.y = round(y0 - 1000 * (cos(theta)));
        circle(rawImage,Point(x0,y0), 4 , Vec3b(100,100,0), 2, 8,0);
        line(rawImage, a, b, Scalar(0,0,255), 1, 8);
        cout << theta*180/M_PI << endl;
    }
    imshow("raw", rawImage);
}

void houghTransform(Mat& inputImage, Mat& rawImage, int threshold)
{
    int N = inputImage.rows;
    int M = inputImage.cols;
    int width = sqrt(pow(N,2)+pow(M,2));
    int height = 180;
    Mat accumulator = Mat::zeros(height, width*2, CV_32FC1);
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<M; j++)
        {
            if(inputImage.at<uchar>(i,j)!=0)
            {
                for(int th=0; th<180; th++)
                {
                    double rho = j*cos((th)*M_PI/180)+i*sin((th)*M_PI/180);
                    accumulator.at<float>(th,round(rho+width))++;
                }
            }
        }
    }
    Mat parameters(accumulator.size(), CV_8UC1);
    normalize(accumulator, parameters, 0, 255, NORM_MINMAX);
    imwrite("parametri.png", parameters);
    vector<Point2d> rette;
    for(int i=0; i<accumulator.rows; i++)
    {
        for(int j=0; j<accumulator.cols; j++)
        {
            if(accumulator.at<float>(i,j)>threshold)
            {
                double rho = j - width;
                double theta = i*M_PI/180;
                rette.push_back(Point2d(theta,rho));
            }
        }
    }
    drawLines(rawImage, rette);
}

int main(int argc, char** argv )
{
    const char* fileName = "recinto.jpg";
    Mat rawImage = imread(fileName, IMREAD_GRAYSCALE);                          //immagine presa in input
    Mat edgesImage(rawImage.size(), CV_8UC1);
    if (!rawImage.data)
    {
        printf("No image data \n");
        return -1;
    }
    blur(rawImage,rawImage, Size(5,5), Point(-1,-1), 4);
    Canny(rawImage,edgesImage,100,150,3);
    //imshow("canny", edgesImage);
    houghTransform(edgesImage, rawImage,120);
    waitKey(0);
    return 0;
}
