#pragma once
#define gaussKernel(x) (2*exp(-x/2))
#define gaussKernelDev(x) (2*exp(-x/2)/2)
#define depth2worldx(fx,cx,depthx,depth) ((depthx - cx) * depth / fx)
#define depth2worldy(fy,cy,depthy,depth) ((depthy - cy) * depth / fy)
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include <fstream>
#include <vector>
#include <queue>
#include "math.h"
#include "time.h"
//#include <algorithm.h>
//#include "OpenNI.h"
using namespace cv;
//using namespace openni;
typedef struct myPoint
{
	double x;
	double y;
	double z;
	double h;
	double s;
	double v;
};
typedef struct Cluster
{
	Point3d center;
	int pt_num;
	double score;
};
class Tracker
{
private:
	string file_path;
	Mat target_hist;
	double radius;
	Mat feature_weight;
	std::vector<Cluster> det_clusters;
	std::vector<Point3d> pts3;
	std::vector<Rect> rects;
	std::vector<double> depths;
	std::vector<double> cam_param;
	std::vector<int> tar_cn;
	std::vector<int> con_cn;
	std::vector<double> crc;
	std::vector<double> crt;
	Point3d occ_pt;
	int hbins;
	int sbins;
	int vbins;
	//double weight;	
	double bandwidth;
	int hx;
	int hy;
	Rect init_rect;
	bool isocc;
	bool islost;
	std::vector<int> status;
	int iter;
	int occ_cn;
public:
	Tracker(void);
	int get_iter()
	{
		return iter;
	};
	Point3d get_curpos()
	{
		return pts3[pts3.size()-1];
	};
	bool is_occluded()
	{
		return isocc;
	};
	Rect get_cur_rect()
	{
		if(rects.size()>=0)
			return rects[rects.size()-1];
	}
	bool get_rect_pos3(const std::vector<double> param, const Mat& depthimg, Rect rect, Point3d& pt);
	bool search_lost(const std::vector<double> param, const Mat& depthimg, const Mat& hsv, Rect rect, 
		Point3d pt0, Point3d& pt1);
	void get_clusters(const std::vector<double> param, const Mat& depthimg, const Mat& hsv,
		Point3d pt0, std::vector<Cluster>& clusters);
	void initialize(const std::vector<double> param,const Mat& depthimg, const Mat& bgrimg, 
		int h_bins, int s_bins, int v_bins, Rect rect);
	void track_all();
	void track(const std::vector<double> param, const Mat& depthimg, const Mat& hsv, 
		const Mat& depthimg_pre, const Mat& hsv_pre);
	void track_ms3d(const std::vector<double> param, const Mat& depthimg, const Mat& hsv,
		const Mat& depthimg_pre, const Mat& hsv_pre);
	void track_with_detector(const std::vector<double> param, const Mat& depthimg, const Mat& hsv,
		const Mat& depthimg_pre, const Mat& hsv_pre);
	void tracker_adaption(const Mat& hist1);
	void cal_hist(const std::vector<double> param, const Mat& hsv, const Mat& depthimg, Mat& hist, Rect br, 
		std::vector<int>& ref, std::vector<myPoint>& pts, Point3d pt, Point3d& center, int& pt_cn);
	void cal_hist(const std::vector<double> param, const Mat& hsv, const Mat& depthimg, 
		Mat& hist, Rect br, Point3d pt );
	
	void showPdf2d(const Mat& hsv);
	bool ms3d(const std::vector<double> param, const Mat& depthimg,
		const Mat& hsv, const Mat& hist0, Mat& hist1, Point3d pt0, Point3d& pt1);

	void computeContext(const std::vector<double> param, const Mat& depthimg,
		const Mat& bgrimg, Point3d pt);
	void weightFeatures(const std::vector<double> param,const Mat& depthimg, 
				const Mat& bgrimg, Rect br, Point3d pt);
	
	void calBondingRect1(const std::vector<double> param,Point3d pt, Rect& br);
	void calBondingRect(const std::vector<double> param,Point3d pt, Rect& br);

	void depthToWorld(const std::vector<double> param,double depthx,double depthy,double depth,double & wx,double & wy,double & wz);
	void worldToDepth(const std::vector<double> param,double& depthx,double& depthy,double& depth,double wx,double wy,double wz);
	void showTarget(const std::vector<double> param,Mat& img,const Mat& depthimg, Rect& br, Point3d pt);

	void saveTraj(char* filename);
	void saveTrajBox(char* filename);
	void readTraj(std::vector<Point3d>& pts, char* filename, double & radius);

	~Tracker(void);
};
double bhattaCoeff(const Mat& hist1, const Mat& hist2);
void saveMatrix(char* filename, const Mat & m);
double ptDist(const Point3d& pt1, const Point3d& pt2);

