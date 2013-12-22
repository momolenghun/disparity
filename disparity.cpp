#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include "maxflow/graph.h"

#include "image.h"

#define SQR(x) ((x)*(x))
#define NCC_NGH_SIZE 5

Image<float> GraphCutLabeling(const Image<float> &I1, const Image<float> &I2, int minDisparity, int maxDisparity, float lambda);

void readme() {
	printf("Usage: disparity left_image right_image num_disparity lambda\n");
}

// Sauve un maillage triangulaire dans un fichier ply.
// v: sommets (x,y,z), f: faces (indices des sommets), col: couleurs (par sommet) 
bool savePly(const string& name,const vector<Point3f>& v, const vector<Vec3i>& f,const vector<Vec3b>& col) {
	assert(v.size()==col.size());
	ofstream out(name.c_str());
	if (!out.is_open()) {
		cout << "Cannot save " << name << endl;
		return false;
	}
	out << "ply" << endl
		<< "format ascii 1.0" << endl
		<< "element vertex " << v.size() << endl
		<< "property float x" << endl
		<< "property float y" << endl
		<< "property float z" << endl
		<< "property uchar red" << endl                   
		<< "property uchar green" << endl
		<< "property uchar blue" << endl
		<< "element face " << f.size() << endl
		<< "property list uchar int vertex_index" << endl  
		<< "end_header" << endl;
	for (int i=0;i<v.size();i++)
		out << v[i].x << " " << v[i].y << " " << v[i].z << " " << int(col[i][2]) << " " << int(col[i][1]) << " " << int(col[i][0]) << " " << endl;
	for (int i=0;i<f.size();i++)
		out << "3 " << f[i][0] << " " << f[i][1] << " " << f[i][2] << endl;
	out.close();
	return true;
}

// Builds the mesh based on the correspondence found

void buildMesh(const Image<Vec3b>& I, const Image<float>& disparity, float C1, float C2) {

	vector<Point3f> verts;
	vector<Vec3b> colors;
	vector<Vec3i> faces;

	for(int i = 0; i < disparity.cols; i++) {
		for(int j = 0; j < disparity.rows; j++) {
			verts.push_back(Point3f(i, j, C1/(C2 + disparity(i,j))));
			colors.push_back(I(i,j));
		}
	}

	// Creates a grid like this:
	// x--x--x-
	// | /| /|
	// |/ |/ |  ...
	// x--x--x-
	// | /| /|
	//    .
	//    .
	//    .

	for(int i = 0; i < disparity.cols-1; i++) {
		for(int j = 0; j < disparity.rows-1; j++) {
			Vec3i fa, fb;
			fa[0] = i*disparity.rows + j; fa[1] = i*disparity.rows + j + 1; fa[2] = (i+1)*disparity.rows + j;
			fb[0] = i*disparity.rows + j + 1; fb[1] = (i+1)*disparity.rows + j; fb[2] = (i+1)*disparity.rows + j + 1;
			faces.push_back(fa); faces.push_back(fb);
		}
	}

	savePly("visage.ply", verts, faces, colors);
}

Image<float> loadGrayscaleImage(char *image) {
	Mat img = imread( image ), imgGray;
	Image<float> ret;

	cvtColor(img, imgGray, CV_BGR2GRAY);
	imgGray.convertTo(ret, CV_32F);

	imshow(image, img);

	return ret;
}

int main(int argc, char** argv)
{
	if(argc != 6) {
		readme();
		return 1;
	}

	int minDisparity, maxDisparity;
	float lambda;

	sscanf(argv[3], "%d", &minDisparity);
	sscanf(argv[4], "%d", &maxDisparity);
	sscanf(argv[5], "%f", &lambda);

	printf("%s\n", argv[1]);

	Image<float> leftImage = loadGrayscaleImage( argv[1] );
	Image<float> rightImage = loadGrayscaleImage( argv[2] );

	Image<float> initialSolution = GraphCutLabeling(leftImage, rightImage, minDisparity, maxDisparity, lambda);

	buildMesh(imread(argv[1]), initialSolution, 100000, 100);

	imshow("Initial solution", initialSolution.greyImage());
	waitKey();
	//Image<float> finalSolution = AlphaExpansion(img_1, img_2, initialSolution);
}

// Implements the solution to the multi-labeling problem with energy function
// V_{p,q} = \lambda |f_p - f_q|
Image<float> GraphCutLabeling(const Image<float> &left, const Image<float> &right, int minDisparity, int maxDisparity, float lambda) {
	printf("%d %f\n", maxDisparity, lambda);
	int disparitySpan = maxDisparity-minDisparity;
	Image<float> leftMean = meanImage(left, NCC_NGH_SIZE), rightMean = meanImage(right, NCC_NGH_SIZE);
	Image<float> result(left.cols-maxDisparity, left.rows, CV_32F);
	int resultWidth = result.cols;

	Graph<float, float, float> G(left.rows * resultWidth * disparitySpan,  left.rows * resultWidth * disparitySpan);
	G.add_node(left.rows * resultWidth * disparitySpan);

	for(int i = 0; i < left.rows; i++)
		for(int j = 0; j < resultWidth; j++) {
			int baseVertex = (i*resultWidth + j)*disparitySpan;

			G.add_tweights(baseVertex, SQR(1-NCC(left, leftMean, Point(j,i), right, rightMean, Point(j+minDisparity,i), NCC_NGH_SIZE)),0);
			for(int k = 0; k < disparitySpan-1; k++)
				G.add_edge(baseVertex + k, baseVertex + k + 1,  SQR(1-NCC(left, leftMean, Point(j,i), right, rightMean, Point(j+k+minDisparity+1,i), NCC_NGH_SIZE)), 3000000);
			G.add_tweights(baseVertex + disparitySpan - 1, 0, SQR(1-NCC(left, leftMean, Point(j,i), right, rightMean, Point(j+maxDisparity,i), NCC_NGH_SIZE)));

			for(int di = 0; di <= 1; di++)
				for(int dj = 0; dj <= 1; dj++) {
					if(di + dj == 0 || di * dj == 1) continue;
					int ii = i + di;
					int jj = j + dj;
					if(ii < left.rows && jj < resultWidth) {
						int baseVertexii = (ii*resultWidth + jj)*disparitySpan;
						for(int k = 0; k < disparitySpan; k++)
							G.add_edge(baseVertex + k, baseVertexii + k, lambda, lambda);
					}
				}
		}

	G.maxflow();

	int minFound = 100, maxFound = 0;

	for(int i = 0; i < left.rows; i++)
		for(int j = 0; j < resultWidth; j++) {
			int baseVertex = (i*resultWidth + j)*disparitySpan;
			bool hasMaxDisparity = true;
			for(int k = 0; k < disparitySpan; k++) {
				if(G.what_segment(baseVertex + k) == Graph<float,float,float>::SINK) {
					result(j,i) = k+minDisparity;
					hasMaxDisparity = false;
					break;
				}
			}
			if(hasMaxDisparity) result(j,i) = maxDisparity;
			minFound = min(minFound, (int)result(j,i));
			maxFound = max(maxFound, (int)result(j,i));
		}

	printf("%d %d\n", minFound, maxFound);

	return result;
}

Image<float> AlphaExpand(const Image<float> &I1, const Image<float> &I2,
		const Image<float> &curSolution, int alpha) {
}

// Alpha-expansion on a graph with a given initial solution and energy function
// V_{p,q} = min(\lambda |f_p - f_q|, k)
Image<float> AlphaExpansion(const Image<float> &I1, const Image<float> &I2,
		Image<float> &initialSolution, int numIterations = 3) {
	if(numIterations == 0) return initialSolution;

}
