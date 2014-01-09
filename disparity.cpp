#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>

#include "disparity.h"

// Implements the solution to the multi-labeling problem with energy function
// V_{p,q} = \lambda |f_p - f_q|
Image<int> Disparity::GraphCutLabeling() {
	Image<int> result(left.cols-maxDisparity, left.rows, CV_32S);
	int resultWidth = result.cols;

	Graph<float, float, float> G(left.rows * resultWidth * disparitySpan,  left.rows * resultWidth * disparitySpan);
	G.add_node(left.rows * resultWidth * disparitySpan);

	printf("Building graph\n");
	for(int i = 0; i < left.rows; i++)
		for(int j = 0; j < resultWidth; j++) {
			int baseVertex = (i*resultWidth + j)*disparitySpan;

			G.add_tweights(baseVertex, VertexWeight(i, j, 0),0);
			for(int k = 0; k < disparitySpan-1; k++)
				G.add_edge(baseVertex + k, baseVertex + k + 1, VertexWeight(i, j, k+1), 1e9);
			G.add_tweights(baseVertex + disparitySpan - 1, 0, VertexWeight(i, j, disparitySpan));

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

	printf("Computing maxflow\n");
	G.maxflow();

	int minFound = maxDisparity, maxFound = minDisparity;

	for(int i = 0; i < left.rows; i++)
		for(int j = 0; j < resultWidth; j++) {
			int baseVertex = (i*resultWidth + j)*disparitySpan;

			result(j,i) = maxDisparity;
			for(int k = 0; k < disparitySpan; k++) {
				if(G.what_segment(baseVertex + k) == Graph<float,float,float>::SINK) {
					result(j,i) = k+minDisparity;
					break;
				}
			}

			minFound = min(minFound, (int)result(j,i));
			maxFound = max(maxFound, (int)result(j,i));
		}

	printf("Minimum disparity found: %d\n", minFound);
	printf("Maximum disparity found: %d\n", maxFound);

	return result;
}

Image<int> Disparity::AlphaExpand(const Image<int> &curSolution, int alpha) {
	int disparityThreshold = 5;

	Graph<float, float, float> G(curSolution.rows * curSolution.cols, curSolution.rows * curSolution.cols);
	G.add_node(curSolution.rows * curSolution.cols);

	Image<int> ret = curSolution.clone();

	for(int i = 0; i < curSolution.rows; i++)
		for(int j = 0; j < curSolution.cols; j++) {
			int baseVertex = i*curSolution.cols + j; 
			if(curSolution(j,i) == alpha)
				G.add_tweights(baseVertex, VertexWeight(i, j, alpha-minDisparity), VertexWeight(i, j, 1e9));
			else G.add_tweights(baseVertex, VertexWeight(i, j, alpha-minDisparity), VertexWeight(i, j, curSolution(j,i)-minDisparity));

			for(int di = 0; di <= 1; di++)
				for(int dj = 0; dj <= 1; dj++) {
					if(di + dj == 0 || di * dj == 1) continue;
					int ii = i + di;
					int jj = j + dj;
					if(ii < curSolution.rows && jj < curSolution.cols) {
						int baseVertexii = ii*curSolution.cols + jj;
						if(curSolution(jj, ii) == curSolution(j, i))
							G.add_edge(baseVertex, baseVertexii,
								lambda*min(disparityThreshold, abs(alpha - curSolution(jj, ii))),
								lambda*min(disparityThreshold, abs(alpha - curSolution(j, i))));
						else {
							int intermediate = G.add_node(1);
							G.add_edge(baseVertex, intermediate,
									lambda*min(disparityThreshold, abs(alpha - curSolution(j, i))),
									lambda*min(disparityThreshold, abs(alpha - curSolution(j, i))));
							G.add_edge(intermediate, baseVertexii,
									lambda*min(disparityThreshold, abs(alpha - curSolution(jj, ii))),
									lambda*min(disparityThreshold, abs(alpha - curSolution(jj, ii))));
							G.add_tweights(intermediate, 0, 
									lambda*min(disparityThreshold, abs(curSolution(j,i) - curSolution(jj, ii))));
						}
					}
				}
		}
	
	G.maxflow();
	
	for(int i = 0; i < ret.rows; i++)
		for(int j = 0; j < ret.cols; j++)
			if(G.what_segment(i*ret.cols + j) == Graph<float,float,float>::SINK)
				ret(j,i) = alpha;

	return ret;
}

// Alpha-expansion on a graph with a given initial solution and energy function
// V_{p,q} = min(\lambda |f_p - f_q|, k)
Image<int> Disparity::AlphaExpansion(const Image<int> &initialSolution, int numIterations) {
	Image<int> finalSolution = initialSolution.clone();

	printf("Doing one iteration of the alpha expansion\n");
	for(int i = 0; i < numIterations; i++)
		for(int alpha = maxDisparity; alpha >= minDisparity; alpha--)
			finalSolution = AlphaExpand(finalSolution, alpha);

	return finalSolution;
}
