#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#define LEN 5 //KERNEL DIMENSION FOR GAUSSIAN SMOOTHING
#define PAD floor(LEN/2)
Mat gn;
float th,tl;
Mat vertSobel = (Mat_<char>(3,3) <<-1, -2, -1,
                                    0,  0,  0,
                                    1,  2,  1);
Mat horizSobel = (Mat_<char>(3,3) <<-1, 0, 1,
                                    -2, 0, 2,
                                    -1, 0, 1);

void recursivePromotion(Mat& gnh, Mat& gnl, int i, int j)
{
    int N=gnh.rows;
    int M=gnl.cols;
    if(i!=0 && j!=0 && i<N-(2*PAD) && j<M-(2*PAD))
    {
        for(int k=-1; k<=1; k++)
            for(int l=-1; l<=1; l++)
            {
                if(gnl.at<float>(i+k,j+l)!=0)//IF CONNECTED PROMOTE A WEAK EDGE TO STRONG
                {
                    gnh.at<float>(i+k,j+l) = 255;
                    gnl.at<float>(i+k,j+l) = 0;                    
                    recursivePromotion(gnh,gnl,i+k,j+l);
                }
            }
    }
}

void hysteresis(const Mat& gn)
{
    Mat gnh = Mat::zeros(gn.size(), gn.type());
    Mat gnl = Mat::zeros(gn.size(), gn.type());
    Mat output(gn.size(),CV_8UC1);
    int N = gn.rows;
    int M = gn.cols;
    // cout << "Th: " <<th << endl << "Tl: " << tl << endl << "------------------" << endl;
    //APPLY THRESHOLDS, OBTAINING GNH(strong edges) AND GNL(weak edges)
    for(int i=PAD; i<N-PAD; i++)
    {
        for(int j=PAD; j<M-PAD; j++)
        {
            if(gn.at<float>(i,j) >= th)
                gnh.at<float>(i,j) = 255;//STRONG EDGES
            if(gn.at<float>(i,j) > tl && gn.at<float>(i,j) < th)
                gnl.at<float>(i,j) = 255;//WEAK EDGES
        }
    }
    // imshow("Strong edges", gnh);
    // imshow("Weak edges", gnl);    
    for(int i=PAD; i<N-PAD; i++)
    {
        for(int j=PAD; j<M-PAD; j++)
        {
            if(gnh.at<float>(i,j) != 0) //ONLY VISIT STRONG EDGES
            {
                //8-CONNECTIVITY
                for(int k=-1; k<=1; k++)
                    for(int l=-1; l<=1; l++)
                        if(gnl.at<float>(i+k,j+l)!=0)//IF CONNECTED PROMOTE A WEAK EDGE TO STRONG
                        {
                            gnh.at<float>(i+k,j+l) = 255;
                            gnl.at<float>(i+k,j+l) = 0;
                            recursivePromotion(gnh,gnl,i+k,j+l);
                        }
            }
            output.at<uchar>(i,j) = round(gnh.at<float>(i,j));
        }
    }
    Mat outputImageNoPad;
    //output(Rect(Point(PAD, output.cols-PAD), Point(output.rows-PAD, PAD))).copyTo(outputImageNoPad);
    imshow("Canny", output);
    imwrite("canny.jpg", output);
}

void nonMaximaSuppression(Mat& gn, const Mat& magnitudo, const Mat& alfa)
{
    int N = gn.rows;
    int M = gn.cols;
    short dir;
    float a;
    // Mat x (gn.size(), CV_8UC1);
    // Mat y (gn.size(), CV_8UC1);
    for(int i=PAD; i<N-PAD; i++)
    {
        for(int j=PAD; j<M-PAD; j++)
        {
            a = alfa.at<float>(i,j);
            //NORMALIZZIAMO (SU 4) LA DIREZIONE DEL GRADIENTE, CHE RAPPRESENTA LA DIREZIONE DELL'EDGE
            //IN SENSO ANTI-ORARIO 0=HOR, 45=DIAG, 90=VERT, 135=DIAG
            if(-22.5 <= a <= 22.5 || 157.5 <= a <= 180 || -180 <= a <= -157.5)  //edge orizzontale -> direzione del gradiente verticale
                dir = 0;
            else if(22.5 <= a <= 67.5 || -157.5 <= a <= -112.5)                 //edge diagonale -> direzione del gradiente diagonale opposta
                dir = 45;
            else if(67.5 <= a <= 112.5 || -112.5 <= a <= -67.5)                 //edge verticale -> direzione del gradiente orizzontale
                dir = 90;
            else if(-67.5 <= a <= -22.5 || 112.5 <= a <= 157.5)                 //edge diagonale -> direzione del gradiente diagonale opposta
                dir = 135;
            //SOPPRESSIONE DEI NON MASSIMI
            if(dir==0)//direzione del gradiente orizzontale - edge verticale
                if(magnitudo.at<float>(i,j)>=magnitudo.at<float>(i,j-1) && magnitudo.at<float>(i,j)>=magnitudo.at<float>(i,j+1))
                        gn.at<float>(i,j) = magnitudo.at<float>(i,j);
            else if(dir==45)//edge diagonale
                if(magnitudo.at<float>(i,j)>=magnitudo.at<float>(i-1,j+1) && magnitudo.at<float>(i,j)>=magnitudo.at<float>(i+1,j-1))
                    gn.at<float>(i,j) = magnitudo.at<float>(i,j);
            else if(dir==90)//direazione del gradiente verticale - edge orizzontale
                if(magnitudo.at<float>(i,j)>=magnitudo.at<float>(i-1,j) && magnitudo.at<float>(i,j)>=magnitudo.at<float>(i+1,j))
                    gn.at<float>(i,j) = magnitudo.at<float>(i,j);
            else if(dir==135)//edge diagonale
                if(magnitudo.at<float>(i,j)>=magnitudo.at<float>(i-1,j-1) && magnitudo.at<float>(i,j)>=magnitudo.at<float>(i+1,j+1))
                    gn.at<float>(i,j) = magnitudo.at<float>(i,j);
            // x.at<uchar>(i,j) = round(gn.at<float>(i,j));
            // y.at<uchar>(i,j) = round(magnitudo.at<float>(i,j));
        }
    }
    // imshow("non maxima", x);
    // imshow("magnitudo", y);
}

void sobel(const Mat& inputImage, Mat& gx, Mat& gy, Mat& alfa, Mat& magnitudo)
{
    int N = inputImage.rows;
    int M = inputImage.cols;
    int xGradient, yGradient;
    for(int i=PAD; i<N-PAD; i++)
    {
        for(int j=PAD; j<M-PAD; j++)
        {
            xGradient = 0, yGradient = 0;
            for(int k=0; k<3; k++)
            {
                for(int l=0; l<3; l++)
                {
                    xGradient += inputImage.at<uchar>(i+k-1,j+l-1)*horizSobel.at<char>(k,l);
                    yGradient += inputImage.at<uchar>(i+k-1,j+l-1)*vertSobel.at<char>(k,l);
                }
            }
            xGradient = abs(xGradient);
            yGradient = abs(yGradient);
            gx.at<uchar>(i,j) = xGradient > 255? 255: xGradient;
            gy.at<uchar>(i,j) = yGradient > 255? 255: yGradient;
            alfa.at<float>(i,j) = (atan2(yGradient,xGradient)*180)/M_PI;
            magnitudo.at<float>(i,j) = xGradient + yGradient;
        }
    }
    //imshow("gx", gx);
    //imshow("gy", gy);
    //imshow("magnitudo", magnitudo);
    //imshow("alfa", alfa);
}

void gaussianFilter(const Mat& inputImage, Mat &outputImage)
{
    Mat kernel = Mat::zeros(LEN, LEN, CV_32FC1);
    int N = inputImage.rows;
    int M = inputImage.cols;
    short var = 3;
    //POPOLO IL KERNEL GAUSSIANO
    for(int i=-PAD; i<=PAD; i++)
        for(int j=-PAD; j<=PAD; j++)
            kernel.at<float>(i+PAD,j+PAD) = exp(-((pow(i,2)/(2*var)) + (pow(j,2)/(2*var))));
    float min = kernel.at<float>(0,0);
    float B = 1/min;
    for(int i=-PAD; i<=PAD; i++)
        for(int j=-PAD; j<=PAD; j++)
            kernel.at<float>(i+PAD,j+PAD) = round(kernel.at<float>(i+PAD,j+PAD)*B);
    float sum = 0;
    for(int i=-PAD; i<=PAD; i++)
        for(int j=-PAD; j<=PAD; j++)
            sum += kernel.at<float>(i+PAD,j+PAD);
    //APPLICO LO SMOOTHING GAUSSIANO
    float count;
    for(int i=PAD; i<=N-PAD; i++)
    {
        for(int j=PAD; j<= M-PAD; j++)
        {
            count = 0;
            for(int k=-PAD; k<=PAD; k++)
            {
                for(int l=-PAD; l<=PAD; l++)
                {
                     count += (int)inputImage.at<uchar>(i+k,j+l)*kernel.at<float>(k+PAD,l+PAD);
                }
                outputImage.at<uchar>(i,j) = round(count/sum); //normalizzo
            }
        }
    }
    //imshow("smoothed image", outputImage);
}

void frameMatrix(const Mat& rawImage, Mat& outputImage)
{
    int N = rawImage.rows;
    int M = rawImage.cols;
    outputImage = Mat::zeros(N+2*PAD,M+2*PAD, rawImage.type());  
    for(int i=PAD; i<N; i++)
    {
        for(int j=PAD; j<M; j++)
        {
            outputImage.at<uchar>(i,j) = rawImage.at<uchar>(i-PAD,j-PAD);
        }
    }
    //imshow("framed image", image);
}

void minThreshold(int selectedTreshold, void*)
{
    tl = selectedTreshold;
    hysteresis(gn);
}

void maxThreshold(int selectedTreshold, void*)
{
    th = selectedTreshold;
    hysteresis(gn);
}


void iterativeHysteresis()
{
    int threshold;
    createTrackbar("Th:", "Canny", &threshold, 255, maxThreshold);
    createTrackbar("Tl:", "Canny", &threshold, 255, minThreshold);
}

int main(int argc, char** argv )
{
    const char* fileName = "lena.png";
    Mat rawImage = imread(fileName, IMREAD_GRAYSCALE);                          //immagine presa in input
    if (!rawImage.data)
    {
        printf("No image data \n");
        return -1;
    }
    Mat image;            
    namedWindow("Canny", WINDOW_AUTOSIZE); 
    imshow("Canny", rawImage);                                        
    frameMatrix(rawImage,image);
    Mat outputImage = Mat(image.size(),image.type());                           //immagine di output
    Mat gx = Mat::zeros(image.rows, image.cols, rawImage.type());               //gradiente orizzontale
    Mat gy = Mat::zeros(image.rows, image.cols, rawImage.type());               //gradiente verticale
    Mat alfa = Mat::zeros(image.rows, image.cols, CV_32FC1);                    //direzione del gradiente
    Mat magnitudo = Mat::zeros(image.rows, image.cols, CV_32FC1);               //magnitudo del gradiente
    gn = Mat::zeros(image.rows, image.cols, CV_32FC1);                          //immagine di edge assottigliati
    gaussianFilter(image, outputImage);
    sobel(outputImage,gx,gy,alfa,magnitudo);    //passo l'immagine blurrata (outputImage) e le matrici da riempire
    nonMaximaSuppression(gn,magnitudo,alfa);    //passo la matrice da riempire che conterrà gli edge assottigliati (gn), magnitudo e direzione gradiente
	iterativeHysteresis();
    Mat testImage = Mat(image.size(), image.type());
    Canny(image,testImage, 34, 47);
    imshow("opencv",testImage);
    waitKey(0);
    return 0;
}
