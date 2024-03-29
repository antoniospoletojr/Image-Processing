#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>

#define DELTA(size) round(size / 5)
#define LOWER_DELTA_THRESHOLD 17 //se aumenta ottengo meno clusters perchè stoppo prima il taglio degli istogrammi

using namespace std;
using namespace cv;

Mat rawImage, blurredImage, histograms[3];
vector<Mat> finalClusters;

class Sample
{
    public:
        int occurrence;   //y dell'istogramma
        int chromaticity; //x dell'istogramma

        Sample() {}
        Sample(int occurrence, int chromaticity)
        {
            this->occurrence = occurrence;
            this->chromaticity = chromaticity;
        }
        bool operator<(const Sample &nextSample) const
        {
            return (occurrence > nextSample.occurrence);
        }
};

class OhlanderCore
{
    public:
        vector<Sample> channels[3];
        int deltas[3];
        int valley;
        int selectedChannel;

        OhlanderCore(Mat mask)
        {
            initHist(mask);
            selectedChannel = -1;
            valley = INT_MAX;
            //popola i canali (BGR) con dei Sample (memorizzo in esso la sua cromaticità e la sua occorrenza)
            for (int c = 0; c < 3; c++)
                for (int i = 0; i < histograms[c].rows; i++)
                    channels[c].push_back(Sample((int)histograms[c].at<float>(i, 0), i));
            calcDeltas();
            for (int c = 0; c < 3; c++)
                sort(channels[c].begin(), channels[c].end());
        }

        void initHist(Mat mask)
        {
            Mat bgrPlanes[3];
            split(blurredImage, bgrPlanes);
            /// Establish the number of bins
            int histSize = 256;
            /// Set the ranges ( for B,G,R) )
            float range[] = {0, 256};
            const float *histRange = {range};
            /// Compute the histograms:
            calcHist(&bgrPlanes[0], 1, 0, mask, histograms[0], 1, &histSize, &histRange);
            calcHist(&bgrPlanes[1], 1, 0, mask, histograms[1], 1, &histSize, &histRange);
            calcHist(&bgrPlanes[2], 1, 0, mask, histograms[2], 1, &histSize, &histRange);
        }

        void calcDeltas() //calcola l'estensione corrente dell'istogramma
        {
            int leftIndex, rightIndex;
            for (int c = 0; c < 3; c++)
            {
                for (int l = 0; l < histograms[c].rows; l++)
                {
                    if (histograms[c].at<float>(l, 0) != 0)
                    {
                        leftIndex = l;
                        break;
                    }
                }
                for (int r = histograms[c].rows - 1; r >= 0; r--)
                {
                    if (histograms[c].at<float>(r, 0) != 0)
                    {
                        rightIndex = r;
                        break;
                    }
                }
                deltas[c] = DELTA((rightIndex - leftIndex));
            }
        }

        void tryClustering()
        {
            pair<Sample, Sample> hills[3];
            Sample max, secondMax;
            //determina i picchi (hills) di ogni canale
            for (int c = 0; c < 3; c++)
            {
                max = channels[c].at(0);
                secondMax = max;
                for (int i = 1; i < channels[c].size(); i++)
                {
                    //se il picco successivo dista DELTA dal massimo, allora consideralo come secondo massimo
                    if (abs(max.chromaticity - channels[c].at(i).chromaticity) > deltas[c])
                    {
                        secondMax = channels[c].at(i);
                        break;
                    }
                }
                hills[c] = make_pair(max, secondMax);
            }
            //determina quale canale considerare per il clustering (istogramma con picco massimo ed una valle)
            int maxOccurrence = 0;
            for (int c = 0; c < 3; c++)
            {
                //se l'istogramma al canale C ha 2 picchi
                if (hills[c].first.chromaticity != hills[c].second.chromaticity)
                {
                    if (hills[c].first.occurrence > maxOccurrence)
                    {
                        maxOccurrence = hills[c].first.occurrence;
                        selectedChannel = c;
                    }
                }
            }
            if (isClusterizable())
                valley = round(abs(hills[selectedChannel].first.chromaticity + hills[selectedChannel].second.chromaticity) / 2);
        }

        inline bool isClusterizable() { return selectedChannel >= 0 && deltas[selectedChannel] >= LOWER_DELTA_THRESHOLD; }
};

void createMasks(const Mat& currMask, Mat& firstMask, Mat& secondMask, int channel, int chromaThreshold)
{
    firstMask = Mat::zeros(blurredImage.size(), CV_8UC1);
    secondMask = Mat::zeros(blurredImage.size(), CV_8UC1);
    for (int i = 0; i < blurredImage.rows; i++)
    {
        for (int j = 0; j < blurredImage.cols; j++)
        {
            //Se la posizione corrente è attiva (mascherata)
            if (currMask.at<uchar>(i, j) == 1)
            {
                if (blurredImage.at<Vec3b>(i, j)[channel] < chromaThreshold)
                {
                    firstMask.at<uchar>(i, j) = 1;
                    secondMask.at<uchar>(i, j) = 0;
                }
                else
                {
                    firstMask.at<uchar>(i, j) = 0;
                    secondMask.at<uchar>(i, j) = 1;
                }
            }
        }
    }
}

void Ohlander(const Mat mask)
{
    OhlanderCore maskedRegion(mask);
    maskedRegion.tryClustering();
    if (maskedRegion.isClusterizable())
    {
        Mat firstMask, secondMask;
        createMasks(mask, firstMask, secondMask, maskedRegion.selectedChannel, maskedRegion.valley);
        Ohlander(firstMask);
        Ohlander(secondMask);
    }
    else
        finalClusters.push_back(mask);
}

void printOhlander()
{
    cout << "Number of clusters: " << finalClusters.size() << endl;
    srand(time(NULL));
    Mat outputImage = Mat::zeros(blurredImage.size(), blurredImage.type());
    for (int i = 0; i < finalClusters.size(); i++)
    {
        Vec3b color(rand() % 255, rand() % 255, rand() % 255);
        for (int row = 0; row < outputImage.rows; row++)
        {
            for (int col = 0; col < outputImage.cols; col++)
            {
                if (finalClusters.at(i).at<uchar>(row, col) != 0)
                {
                    outputImage.at<Vec3b>(row, col) = color;
                }
            }
        }
    }
    imshow("Ohlander", outputImage);
    waitKey(0);
}

int main(int argc, char **argv)
{
    const char *fileName = "mele.jpg";
    rawImage = imread(fileName, IMREAD_COLOR); //immagine presa in input
    if (rawImage.empty())
    {
        cerr << "--- ERROR --- No data found" << endl;
        exit(EXIT_FAILURE);
    }
    blurredImage = rawImage.clone();
    GaussianBlur(rawImage, blurredImage, Size(5, 5), 8); //più è alta la varianza, più è forte lo smoothing
    Ohlander(Mat::ones(blurredImage.size(), CV_8UC1));
    printOhlander();
    return 0;
}