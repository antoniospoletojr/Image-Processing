#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <stdlib.h>
#include <time.h>  
#define K 6
#define ITERATION_THRESHOLD 8
#define SWAPPINESS_THRESHOLD 0.001
using namespace cv;
using namespace std;

class Pixel
{
    public:
        int row;
        int col;
        Vec3f color;

    Pixel(int row, int col, Vec3b color )
    {
        this->row = row;
        this->col = col;
        this->color = color;
    }
};

class Cluster
{
    public:
        Vec3b mean;
        vector<Pixel> pixels;

    Cluster(Pixel pixel)
    {
        this->pixels.push_back(pixel);
        this->mean = pixels.back().color;
    }

    void addPixel(Pixel pixel)
    {
        pixels.push_back(pixel);
    }

    void removePixel(int index)
    {
        this->pixels.erase(this->pixels.begin()+index);
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

vector<Cluster> clusters;
Mat image;

float euclideanDistance(Vec3b p1, Vec3b p2)
{
    return sqrt(pow(p1[0]-p2[0],2)+pow(p1[1]-p2[1],2)+pow(p1[2]-p2[2],2));
}

void init()
{
    srand(time(NULL));
    int N = image.rows;
    int M = image.cols;
    vector<Pixel> vectorImage;
    //VETTORIALIZZIAMO LA NOSTRA IMMAGINE
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            vectorImage.push_back(Pixel(i,j,image.at<Vec3b>(i,j)));
    //INIZIALIZZIAMO I CLUSTERS
    for(int i=0; i<K; i++)
    {
        int index = rand()%vectorImage.size();
        clusters.push_back(vectorImage.at(index));
        vectorImage.erase(vectorImage.begin()+index);
    }
    //RIEMPIAMO I CLUSTERS
    for(auto pixel: vectorImage)
    {
        float min = MAXFLOAT;
        float distance;
        int index = 0;
        for(int j=0; j<K; j++)
        {
            distance = euclideanDistance(pixel.color, clusters.at(j).mean);
            if(distance<min)
            {
                index = j;
                min = distance;                
            }
        }
        clusters.at(index).addPixel(pixel);
    }
}

float reallocateClusters()
{
    float swappiness=0;
    int N = image.rows;
    int M = image.cols;
    //PER OGNI CLUSTER
    for(int k=0; k<K; k++)
    {
        int clusterSize = clusters.at(k).pixels.size();
        //PER OGNI PIXEL DEL CLUSTER
        for(int i=0; i<clusterSize; i++)
        {
            Pixel current = clusters.at(k).pixels.at(i);
            float min = MAXFLOAT;
            float distance;
            int index = 0;
            //CONFRONTA CON LE MEDIE DEI K CLUSTERS
            for(int j=0; j<K; j++)
            {
               distance = euclideanDistance(current.color, clusters.at(j).mean);
                if(distance<min)
                {
                    index = j;
                    min = distance;                
                }
            }
            if(k!=index) //SE IL PIXEL DEVE ESSERE SPOSTATO IN UN ALTRO CLUSTER 
            {
                clusters.at(k).removePixel(i);
                clusters.at(index).addPixel(current);
                clusterSize--;
                swappiness++;
            }
        }
    }
    return swappiness/(N*M);
}

void assignColors()
{
    for(int i=0; i<K; i++)
    {
        Cluster& current = clusters.at(i);
        for(int j=0; j<current.pixels.size(); j++)
        {
            Point coords(current.pixels.at(j).row,current.pixels.at(j).col);
            image.at<Vec3b>(coords.x,coords.y) = current.mean;
        }
    }
}

void kMeans()
{
    //INIZIALIZZAZIONE
    //UN PIXEL RANDOM VIENE ASSEGNATO AD OGNI CLUSTER
    //TUTTI GLI ALTRI VENGONO POI ASSEGNATI AL CLUSTER PIU' VICINO
    init();
    int iterations = 0;
    float swappiness=1;
    while(iterations<ITERATION_THRESHOLD && swappiness > SWAPPINESS_THRESHOLD)
    {
        iterations++;
        //RICALCOLO MEDIA E MATRICE DI COVARIANZA DEI CLUSTERS
        for(int i=0; i<K; i++)
            clusters.at(i).calculateMean();
        //RIALLOCO I PIXEL NEI GIUSTI CLUSTER
        swappiness = reallocateClusters();
        cout << "iteration:" << iterations << " - " << swappiness << endl;
    }
    assignColors();
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
    kMeans();
    imwrite("output.jpg",image);
    return 0;
}




   
