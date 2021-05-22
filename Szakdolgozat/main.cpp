#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <queue>
using namespace std;
#include "opencv2/opencv.hpp"
#include "Color3d.h"
#include "Grid.h"
const string winname = "Colorization - Chrominance Blending";
bool isPress = false;
cv::Mat gray;
cv::Mat temp;
cv::Mat input;
int brush[3] = {0};
const int dim = 3;
const int dx[4] = {-1, 0, 0, 1};
const int dy[4] = {0, -1, 1, 0};
typedef pair<int, int> Pnt;
typedef pair<pair<double, int>, Pnt> Pair;
double calc_weight(double r) {
	return 1.0 / (pow(abs(r), 3) -7);
}
void colorize() {
	set<Color3d> S;
	input = cv::imread("current/image_Ch_blend.bmp", cv::IMREAD_COLOR);
	temp = cv::imread("current/image_marked.bmp", cv::IMREAD_COLOR);
	const int width  = gray.cols;
	const int height = gray.rows;
	for(int y=0; y<height; y++) {
		for(int x=0; x<width; x++) {
			uchar red   = input.at<uchar>(y, x*dim+2);
			uchar green = input.at<uchar>(y, x*dim+1);
			uchar blue  = input.at<uchar>(y, x*dim+0);
			S.insert(Color3d(red, green, blue));
		}
	}

	vector<Color3d> colors(S.begin(), S.end());
	map<Color3d, int> table;
	for(int i=0; i<colors.size(); i++) {
		table[colors[i]] = i;
	}

	priority_queue<Pair, vector<Pair>, greater<Pair> > que;
	Grid<pair<int, double>, map<int, double> > grid(height, width);
	for(int y=0; y<height; y++) {
		for(int x=0; x<width; x++) {
			uchar red   = input.at<uchar>(y, x*dim+2);
			uchar green = input.at<uchar>(y, x*dim+1);
			uchar blue  = input.at<uchar>(y, x*dim+0);
			if(red | green | blue) {

				int color = table[Color3d(red, green, blue)];
				grid.ptrAt(y, x).insert(make_pair(color, 0.0));
				que.push(make_pair(make_pair(0.0, color), Pnt(x, y)));
			}
		}
	}

	while(!que.empty()) {
		double dist  = que.top().first.first;
		int    color = que.top().first.second;
		Pnt    pt    = que.top().second;
		que.pop();

		for(int k=0; k<4; k++) {
			int nx = pt.first  + dx[k];
			int ny = pt.second + dy[k];
			if(nx >= 0 && ny >= 0 && nx < width && ny < height) {
				double ndist = dist + abs(gray.at<uchar>(pt.second, pt.first) - gray.at<uchar>(ny, nx));
				if(grid.ptrAt(ny, nx).find(color) == grid.ptrAt(ny, nx).end()) {
					if(grid.ptrAt(ny, nx).size() < 3) {
						grid.ptrAt(ny, nx).insert(make_pair(color, ndist));
						que.push(make_pair(make_pair(ndist, color), Pnt(nx, ny)));
					}
				} else {
					if(grid.ptrAt(ny, nx)[color] > ndist) {
						grid.ptrAt(ny, nx)[color] = ndist;
						que.push(make_pair(make_pair(ndist, color), Pnt(nx, ny)));
					}
				}
			}
		}
	}

	cv::Mat out = cv::Mat(gray.size(), CV_8UC3, CV_RGB(0, 0, 0));
	for(int y=0; y<height; y++) {
		for(int x=0; x<width; x++) {
			double weight = 0.0;
			Color3d color(0, 0, 0);
			map<int, double>::iterator it;
			for(it = grid.ptrAt(y, x).begin(); it != grid.ptrAt(y, x).end(); ++it) {
				double w = calc_weight(it->second);
				color = color + colors[it->first].multiply(w);
				weight += w;
			}
			color = color.divide(weight);
			for(int c=0; c<dim; c++) {
				out.at<uchar>(y, x*dim+c) = color.v[dim-c-1];
			}
		}
	}
	cv::cvtColor(out, out, cv::COLOR_BGR2YCrCb);
	for(int y=0; y<height; y++) {
		for(int x=0; x<width; x++) {
			out.at<uchar>(y, x*dim+0) = gray.at<uchar>(y, x);
		}
	}
	cv::cvtColor(out, out, cv::COLOR_YCrCb2BGR);
	cv::imwrite("done/Chrominance_blending_result.bmp", out);
	cv::imshow(winname, out);
}
int main(int argc, char** argv) {
	gray = cv::imread("current/image.bmp", cv::IMREAD_GRAYSCALE);
	if(gray.empty()) {
		cout << "Failed to load image file \"" << argv[1] << "\"";
		return -1;
	}
	input = cv::Mat(gray.size(), CV_8UC3, CV_RGB(0, 0, 0));
	cv::namedWindow(winname);
	cv::cvtColor(gray, temp, cv::COLOR_GRAY2BGR);
	cv::imshow(winname, temp);
	int key = 0;
	colorize();
	while(key != 0x1b) {
		key = cv::waitKey(30);
		if(key == 'c') {
			colorize();
		}
	}
	cv::destroyAllWindows();
}
