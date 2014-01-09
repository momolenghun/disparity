#include <cmath>
#include "GCO/Graph.h"

#include "image.h"

#define SQR(x) ((x)*(x))
#define NCC_NGH_SIZE 5

class Disparity {

	int minDisparity, maxDisparity, disparitySpan;
	Image<float> left, right;
	Image<float> leftMean, rightMean;
	float lambda;

	public:
		Disparity(char *leftPath, char *rightPath, int minDisparity, int maxDisparity, float lambda) :
			minDisparity(minDisparity), maxDisparity(maxDisparity), lambda(lambda) {

			disparitySpan = maxDisparity-minDisparity;

			left = LoadGrayscaleImage(leftPath);
			right = LoadGrayscaleImage(rightPath);

			leftMean = meanImage(left, NCC_NGH_SIZE);
		   	rightMean = meanImage(right, NCC_NGH_SIZE);
		}
		Image<int> GraphCutLabeling();
		Image<int> AlphaExpansion(const Image<int> &initialSolution, int numIterations = 3);


	private:
		Image<int> AlphaExpand(const Image<int> &curSolution, int alpha);
		Image<float> LoadGrayscaleImage(char *image) {
			Mat img = imread( image ), imgGray;
			Image<float> ret;

			cvtColor(img, imgGray, CV_BGR2GRAY);
			imgGray.convertTo(ret, CV_32F);

			imshow(image, img);

			return ret;
		}

		// Cost of atributing disparity k for vertex (i,j)
		float VertexWeight(int i, int j, int k) {
			return sqrt(1-NCC(left, leftMean, Point(j,i), right, rightMean, Point(j+k+minDisparity,i), NCC_NGH_SIZE));
		}
};
