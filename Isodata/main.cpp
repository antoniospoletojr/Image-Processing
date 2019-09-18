#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <time.h>
using namespace cv;
using namespace std;
#define K 3                 //numero di cluster iniziali
#define VAR_THRESHOLD 1000  //se aumenta, diminiuisce il numero di clusters prodotti (ammettiamo clusters più eterogenei)
#define SIZE_THRESHOLD 5    //se aumenta, ammettiamo che un numero maggiore di clusters si fondano
#define SUB_CLUSTERS 3      //nella fase di splitting dividiamo il cluster in SUB_CLUSTERS

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

    Cluster(Pixel pixel)
    {
        this->mean = pixel.color;
    }

    inline void addPixel(Pixel pixel)
    {
        pixels.push_back(pixel);
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

    float getVar()
    {
        Vec3f variance; 
        for(int i=0; i<pixels.size();i++)
        {
            variance[0] += pow(pixels.at(i).color[0] - this->mean[0], 2);
            variance[1] += pow(pixels.at(i).color[1] - this->mean[1], 2);
            variance[2] += pow(pixels.at(i).color[2] - this->mean[2], 2);
        }
        variance[0] = variance[0]/pixels.size();
        variance[1] = variance[1]/pixels.size();
        variance[2] = variance[2]/pixels.size();
        return (variance[0]+variance[1]+variance[2])/3;
    }

    inline int getSize()
    {
        return this->pixels.size();
    }

    inline void color(Mat& image)
    {
        for(auto pixel : this->pixels)
        {
            Point& coords = pixel.coords;
            image.at<Vec3b>(coords) = this->mean;
        }
    }

    inline static float euclideanDistance(Vec3b p1, Vec3b p2)
    {
        return norm(p1,p2);
    }
};

Mat image;
vector<Pixel> pixels;
vector<Cluster> clusters;

void populateClusters(vector<Cluster>& inoutClusters, const vector<Pixel> dataset)
{
    float distance, minDistance;
    int index;
    for(int k=0; k<inoutClusters.size();k++)
    {
        inoutClusters.at(k).pixels.clear();
    }
    for(int i=0; i<dataset.size(); i++)
    {
        minDistance = FLT_MAX;
        for(int j=0; j<inoutClusters.size(); j++)
        {
            distance = Cluster::euclideanDistance(inoutClusters.at(j).mean, dataset.at(i).color);
            if(distance<minDistance)
            {
                minDistance = distance;
                index = j;
            }
        }
        inoutClusters.at(index).addPixel(dataset.at(i));
    }
    for(int k=0; k<inoutClusters.size(); k++)
        inoutClusters.at(k).calculateMean();
}

void init()
{
    int N = image.rows;
    int M = image.cols;
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            pixels.push_back(Pixel(Point(j,i), image.at<Vec3b>(i,j)));
    for(int k=0; k<K; k++)
        clusters.push_back(Cluster(pixels.at((k*(N-1)*(M-1))/K)));
    populateClusters(clusters,pixels);
}

void splitCluster(vector<int>& indexes)
{
    vector<Pixel> dataset;
    vector<Cluster> splittedClusters;
    sort(indexes.begin(),indexes.end(),[](int l, int r){return l>r;});
    //per ogni cluster da splittare
    for(int i=0; i<indexes.size(); i++)
    {
        //prendi indice del cluster da eliminare
        int clusterIndex = indexes.at(i);
        //ricopia tutti i pixel del cluster
        dataset = clusters.at(clusterIndex).pixels;
        //dataset.insert(dataset.end(), tempPixels.begin(), tempPixels.end());
        //cancella il cluster perchè verrà splittato in 3 clusters
        clusters.erase(clusters.begin()+clusterIndex);
        //crea 3 nuovi clusters
        for(int j=0; j<3; j++)
        {
            //setta i centroidi in maniera omogenea
            Pixel& centroid = dataset.at(j*dataset.size()/3);
            //pusha i centroidi in nuovi clusters
            splittedClusters.push_back(centroid);
        }
        populateClusters(splittedClusters,dataset);
        //aggiungiamo i nuovi clusters splittati al vettore di clusters
        clusters.insert(clusters.end(), splittedClusters.begin(), splittedClusters.end());
        splittedClusters.clear();
    }
    indexes.clear();
}

void merge(vector<int>& indexes)
{
    //se i clusters sono dispari, effettua il merge fino al cluster n-1 (cioè mergia a 2 a 2, e lascia l'ultimo penzolante);
    int n = indexes.size();
    if(n%2!=0)
        n--;
    //ordiniamo in senso decrescente per evitare cancellazioni fuori range sul vettore clusters
    sort(indexes.begin(),indexes.end(),[](int l, int r){return l>r;});
    for(int i=0; i<n; i+=2)
    {
        vector<Pixel>& curr = clusters.at(indexes.at(i)).pixels;
        vector<Pixel>& next = clusters.at(indexes.at(i+1)).pixels;
        next.insert(next.end(),curr.begin(),curr.end());
        clusters.erase(clusters.begin()+indexes.at(i));
    }
    indexes.clear();
}

void isodata()
{
    init();
    bool converge;
    vector<int> indexes;
    do
    {
        converge = false;
        for(int k=0; k<clusters.size(); k++)
        {
            if(clusters.at(k).getVar()>VAR_THRESHOLD)
            {
                converge = true;
                //memorizza gli indici di cluster da splittare (varianza troppo grande)
                indexes.push_back(k);
            }
        }
        splitCluster(indexes);
        for(int k=0; k<clusters.size(); k++)
        {
            if(clusters.at(k).getSize()<SIZE_THRESHOLD)
            {
                converge = true;
                indexes.push_back(k);
            }
        }
        merge(indexes);
        for(int k=0; k<clusters.size(); k++)
        {
            clusters.at(k).calculateMean();
        }
    }while(converge);
    cout << "Numero di clusters: " << clusters.size();
    for(auto c: clusters)
    {
        c.color(image);
    }
    imshow("isodata",image);
    imwrite("isodata.jpg",image);
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
    isodata();
    waitKey(0);
    return 0;
}
