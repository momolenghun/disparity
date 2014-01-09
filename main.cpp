#include <cstdio>
#include <iostream>
#include <fstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "disparity.h"

void readme() {
	printf("Usage: disparity left_image right_image min_disparity max_disparity lambda\n");
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

void buildMesh(const Image<Vec3b>& I, const Image<int>& disparity, float C1, float C2) {

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

	savePly("output.ply", verts, faces, colors);
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

	Disparity disparity(argv[1], argv[2], minDisparity, maxDisparity, lambda);

	Image<int> initialSolution = disparity.GraphCutLabeling();

	//Image<int> finalSolution = disparity.AlphaExpansion(initialSolution,2);

	imshow("Initial solution", initialSolution.greyImage());
	//imshow("Final solution", finalSolution.greyImage());
	buildMesh(imread(argv[1]), initialSolution, 100000, 100);

	waitKey();
}
