#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
using namespace cv;
using namespace std;

float euclideanDistance(Vec3b p1, Vec3b p2)
{
    return sqrt(pow(p1[0]-p2[0],2)+pow(p1[1]-p2[1],2)+pow(p1[2]-p2[2],2));
}

class Pixel
{
    public:
        Point coords;
        Vec3b color;

    Pixel(){}
    Pixel(Point p, Vec3b color)
    {
        this->coords = p;
        this->color = color;
    }
};

class Cluster
{
    public:
        Vec3b mean;
        vector<Pixel> pixels;
        pair<Pixel,float> x; //PIXEL PIU' LONTANO DAL CENTRO DEL CLUSTER E RELATIVA DISTANZA

    Cluster(Pixel pixel)
    {
        this->mean = pixel.color;
    }

    void addPixel(Pixel pixel)
    {
        pixels.push_back(pixel);
    }

    void calcMax()
    {
        Pixel max;
        float distance,maxDistance=0;
        for(int i=0; i<pixels.size();i++)
        {   
            distance = euclideanDistance(this->mean, pixels.at(i).color);
            if(distance>maxDistance)
            {
                max = pixels.at(i);
                maxDistance = distance;
            }
        }
        this->x=make_pair(max,maxDistance);
    }

    void calculateMean()
    {
        Vec3f mean;
        for(auto pixel: pixels)
        {
            mean[0]+=pixel.color[0];
            mean[1]+=pixel.color[1];
            mean[2]+=pixel.color[2];
        }
        mean[0] = mean[0]/(pixels.size());
        mean[1] = mean[1]/(pixels.size());
        mean[2] = mean[2]/(pixels.size()); 
        this->mean = mean;
    }
};

Mat image;
Mat resizedImage;
vector<Pixel> pixels;
vector<Cluster> clusters;

void init()
{
    //CERCO LA MAX DISTANZA TRA DUE PIXEL NELL'IMMAGINE RIMPICCIOLITA (più veloce)
    int N = resizedImage.rows;
    int M = resizedImage.cols;
    float distance;
    float max = 0.0;
    Pixel p1,p2;
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            pixels.push_back(Pixel(Point(j,i),resizedImage.at<Vec3b>(i,j)));
    for(int i=0; i<N*M; i++)
    {
        for(int j=i+1; j<N*M; j++)
        { 
            distance = euclideanDistance(pixels.at(i).color,pixels.at(j).color);
            if(distance > max)
            {
                max = distance;
                p1 = pixels.at(i);
                p2 = pixels.at(j);
            }
        }
    }
    //TROVATI I DUE PIXEL A MAX DISTANZA TRA LORO LI USO PER CREARE DUE NUOVI CLUSTERS
    clusters.push_back(Cluster(p1));
    clusters.push_back(Cluster(p2));  
    //QUINDI SVUOTO IL VETTORE DI PIXELS DELL'IMMAGINE PER REINSERIRE QUESTA VOLTA I PIXEL DELL'IMMAGINE ORIGINALE (non rimpicciolita)
    pixels.clear();
    N = image.rows;
    M = image.cols;
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            pixels.push_back(Pixel(Point(j,i),image.at<Vec3b>(i,j)));
}

void reallocateCluster()
{
    int N=image.rows;
    int M=image.cols;
    //svuoto i cluster
    for(int k=0; k<clusters.size(); k++)
        clusters.at(k).pixels.clear();
    //ripopolo i cluster
    for(int i=0; i<N*M; i++)
    {
        float minDistance = FLT_MAX, distance;
        int index;
        for(int k=0; k<clusters.size(); k++)
        {
            distance = euclideanDistance(pixels.at(i).color,clusters.at(k).mean);
            if(distance < minDistance)
            {
                index = k;
                minDistance = distance;
            }
        }
        clusters.at(index).addPixel(pixels.at(i));
    }
    //ricalcolo le medie
    for(int k=0; k<clusters.size(); k++)
        clusters.at(k).calculateMean();
}

void MengHeeHeng()
{
    init();
    bool flag = true;
    while(flag)
    {
        flag = false;
        cout << "number of clusters: " << clusters.size() << endl;
        reallocateCluster();
        float q = 0;
        int count = 0;
        for(int i=0; i<clusters.size(); i++)
            for(int j=i+1; j<clusters.size(); j++)
            {
                q += euclideanDistance(clusters.at(i).mean,clusters.at(j).mean);
                count++;
            }
        q = q/count;
        //D è il max dei max?
        for(int i=0; i<clusters.size(); i++)
        {
            clusters.at(i).calcMax();
            if(clusters.at(i).x.second > (q/2))
            {
                flag = true;
                clusters.push_back(Cluster(clusters.at(i).x.first));
                break;
            }
        }
    }
    //Coloro l'immagine in base ai clusters
    for(int i=0; i<clusters.size(); i++)
    {
        vector<Pixel>& cluster = clusters.at(i).pixels;
        for(int j=0; j<cluster.size(); j++)
        {
            Pixel& pixel = cluster.at(j);
            image.at<Vec3b>(pixel.coords.y, pixel.coords.x) = clusters.at(i).mean;
        }
    }
    imwrite("output.jpg",image);
}

int main(int argc, char** argv )
{
    const char* fileName = "recinto.jpg";
    image = imread(fileName, IMREAD_COLOR);//immagine presa in input
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }
    resizedImage = image.clone();
    resize(image,resizedImage, Size(100,100),0,0, INTER_LINEAR);
    MengHeeHeng();
    return 0;
}




   
