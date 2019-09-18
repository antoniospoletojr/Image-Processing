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

int stdThreshold=20;
int pixelsThreshold=20;
Mat inputImage;
Mat outputImage;
Mat visited;

class Region
{
    public:
        int ID;
        double mean;
        double var;
        double std;
        int n;
        Vec3b color;

    Region()
    {
        static int counter=0;
        counter += 1;
        this->ID = counter;
        this->mean = 0;
        this->var = 0;
        this->std = 0;
        this-> color = Vec3b(rand()%256,rand()%256,rand()%256);
        n=0;
    }

    double getVariance(double x)
    {
        int tempN = n;
        double tempVar = var;
        tempN++;
        tempVar += (tempN-1)*(x-mean)*(x-mean)/(tempN);
        return (tempVar/tempN);
    }

    double getStd(double x)
    {
        int tempN = n;
        double tempVar = var;
        tempN++;
        tempVar += (tempN-1)*(x-mean)*(x-mean)/(tempN);
        return sqrt(tempVar/tempN);
    }

    void addPixel(double x)
    {
        n++;
        if (n==1)
        {
            mean = x;
            var = 0;
        }
        else
        {
            var  += (n-1) * (x-mean)*(x-mean) / (n);
            std = sqrt(var);
            mean += (x-mean) / n;
            
        }
    }
};
vector<Region> regions;

void recursion(int y, int x)
{
    visited.at<float>(y,x) = regions.back().ID;
    outputImage.at<Vec3b>(y,x) = regions.back().color;
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
                    // float std=regions.back().getStd(inputImage.at<uchar>(y+i,x+j));
                    float mean=regions.back().mean;
                    if(abs(mean-inputImage.at<uchar>(y+i,x+j))<50)//testa la varianza
                    {
                        visited.at<float>(y+i, x+j) = regions.back().ID;
                        regions.back().addPixel(inputImage.at<uchar>(y+i,x+j));
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
        regions.back().addPixel(inputImage.at<uchar>(seedY,seedX));
        recursion(seedY, seedX);
        if(regions.back().n<pixelsThreshold)
        {
            for(int j=0; j<N; j++)//scorro le righe
                for(int i=0; i<M; i++)//scorro le colonne
                    if(visited.at<float>(j,i)==regions.back().ID)
                        outputImage.at<Vec3b>(j,i) = Vec3b(0,0,0);
            regions.pop_back();
        }
        for(int j=0; j<N; j++)//scorro le righe
            for(int i=0; i<M; i++)//scorro le colonne
                if(visited.at<float>(j,i)==0)
                {
                    seedY=j;
                    seedX=i;
                }
    }
    imwrite("visited.jpg", outputImage);
    cout << "Numbers of regions: "<< regions.size() << endl;
}

int main(int argc, char** argv )
{
    const char* fileName = "circles.jpg";
    inputImage = imread(fileName, IMREAD_GRAYSCALE);//immagine presa in input
    if (!inputImage.data)
    {
        printf("No image data \n");
        return -1;
    }
    // int dim = MAX(inputImage.rows,inputImage.cols);
    //resize(inputImage, inputImage, Size(300,300),0,0,INTER_LINEAR);
    //GaussianBlur(inputImage,inputImage, Size(5,5), 3, 3, 4);
    medianBlur(inputImage,inputImage,3);
    //threshold(inputImage,inputImage,200,255, THRESH_BINARY);
    //imshow("immagine", inputImage);
    //waitKey(0);
    regionGrowing();
    return 0;
}



    // Region A;
    // for(int i=200; i<205; i++)
    //     A.addPixel(i);
    // cout << A.getStd(205) << endl;
    // vector<int> vettore;
    // srand(time(NULL));
    // for(int i=0; i<5; i++)
    // {
    //     vettore.push_back(i);
    //     cout << vettore.at(i) <<" ";
    // }
    // vettore.push_back(3);
    // vettore.push_back(2);
    // variance(vettore);