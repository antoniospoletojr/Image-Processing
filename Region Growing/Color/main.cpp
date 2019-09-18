#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <stdlib.h>
#include <time.h>  
using namespace cv;
using namespace std;

Mat inputImage;
Mat outputImage;
Mat visited;

class Region
{
    public:
        int ID;
        int n;
        Vec3f mean;
        Vec3f standardDeviation;
        float std;

    Region()
    {
        static int counter=0;
        counter += 1;
        ID = counter;
        n=0;
        standardDeviation=0;
        std=0;
    }

    void addPixel(Vec3b x)
    {
        n++;
        if (n==1)
        {
            mean = x;
        }
        else
        {
            standardDeviation[0]  += (n-1) * (x[0]-mean[0])*(x[0]-mean[0]) / (n);
            standardDeviation[1]  += (n-1) * (x[1]-mean[1])*(x[1]-mean[1]) / (n);
            standardDeviation[2]  += (n-1) * (x[2]-mean[2])*(x[2]-mean[2]) / (n);
            mean[0] += (x[0]-mean[0]) / n;
            mean[1] += (x[1]-mean[1]) / n;
            mean[2] += (x[2]-mean[2]) / n;
            std = (sqrt(standardDeviation[0])+sqrt(standardDeviation[1])+sqrt(standardDeviation[2]))/3;
        }
    }

    bool check(Vec3b x)
    {
        Vec3f temp;
        int tempN = n;
        tempN++;
        temp[0]  += (tempN-1) * (x[0]-mean[0])*(x[0]-mean[0]) / (tempN);
        temp[1]  += (tempN-1) * (x[1]-mean[1])*(x[1]-mean[1]) / (tempN);
        temp[2]  += (tempN-1) * (x[2]-mean[2])*(x[2]-mean[2]) / (tempN);
        float oldStd = std;
        float newStd = (sqrt(temp[0])+sqrt(temp[1])+sqrt(temp[2]))/3;
        if(abs(oldStd-newStd)< 20)
            return true;
        return false;   
    }
};
vector<Region> regions;

bool checkUniformity(Vec3b x)
{
    float deltaB = abs(regions.back().mean[0] - x[0]);
    float deltaG = abs(regions.back().mean[1] - x[1]);
    float deltaR = abs(regions.back().mean[2] - x[2]);
    float delta = (deltaB+deltaG+deltaR)/3;
    if(delta<90)
        return true;
    return false;
}


void recursion(int y, int x)
{
    visited.at<float>(y,x) = regions.back().ID;
    outputImage.at<Vec3b>(y,x) = regions.back().mean;
    int N = inputImage.rows;
    int M = inputImage.cols;
    for(int i=-1; i<=1; i++)
    {
        for(int j=-1; j<=1; j++)
        {
            if((y+i)>=0 && (x+j)>=0 && (y+i)<N && (x+j)<M && (i|j)!=0)//se rientra nei bordi dell'immagine e non è il pixel centrale
            {
                if(visited.at<float>(y+i, x+j)==0)//se non l'ho già visitato
                {
                    if(checkUniformity(inputImage.at<Vec3b>(y+i,x+j)))//testa la varianza
                    {
                        visited.at<float>(y+i, x+j) = regions.back().ID;
                        regions.back().addPixel(inputImage.at<Vec3b>(y+i,x+j));
                        recursion(y+i, x+j);//richiamo su questo pixel
                    }
                }
            }
        }
    }
}

void regionGrowing()
{
    int N = inputImage.rows;
    int M = inputImage.cols;
    visited = Mat::zeros(inputImage.size(), CV_32FC1);
    outputImage = Mat::zeros(inputImage.size(),CV_8UC3);
    srand(time(NULL));
    int seedX = rand()%M; //0-700
    int seedY = rand()%N; //0-200
    while(visited.at<float>(seedY,seedX)==0)
    {
        regions.push_back(Region());
        regions.back().addPixel(inputImage.at<Vec3b>(seedY,seedX));
        recursion(seedY, seedX);
        for(int j=0; j<N; j++)//scorro le righe
            for(int i=0; i<M; i++)//scorro le colonne
                if(visited.at<float>(j,i)==0)
                {
                    seedY=j;
                    seedX=i;
                    break;
                }
    }
    imwrite("visited.jpg", outputImage);
    cout << "Numbers of regions: "<< regions.size() << endl;
}

int main(int argc, char** argv )
{
    const char* fileName = "circles.jpg";
    inputImage = imread(fileName, IMREAD_COLOR);//immagine presa in input
    if (!inputImage.data)
    {
        printf("No image data \n");
        return -1;
    }
    //resize(inputImage, inputImage, Size(300,300),0,0,INTER_LINEAR);
    GaussianBlur(inputImage,inputImage, Size(5,5), 3, 3, 4);
    //medianBlur(inputImage,inputImage,3);
    regionGrowing();
    return 0;
}