#include "Tracker.h"


Tracker::Tracker(void)
{
}

bool Tracker::get_rect_pos3(const std::vector<double> param, const Mat& depthimg, Rect rect, Point3d& pt)
{
	double xsum = 0;
	double ysum = 0;
	double zsum = 0;//中心坐标
	double dsum = 0;
	std::vector<double> zall;
	//std::vector<myPoint> ref_pts;//保存所有点的三维坐标
	double zs = 0;
	for (int i = rect.y; i < rect.y + rect.height; i++)
	for (int j = rect.x; j < rect.x + rect.width; j++)
	{
		if (i >= 0 && i < depthimg.rows&&j >= 0 && j < depthimg.cols)
		{
			double d = depthimg.at<double>(i, j);
			//std::cout<<d<<std::endl;
			if (d != 0/*&&d<pt.z*/)
			{
				zall.push_back(d);
				zs += d;
			}
		}
	}
	if (zall.size() > 100)
	{
		zs = zs / zall.size();
		sort(zall.begin(), zall.end());
		int mid = zall.size() / 2;
		dsum = (zall[mid]);
		depthToWorld(param, rect.x + rect.width / 2, rect.y + rect.height / 2, dsum, xsum, ysum, zsum);
		pt.x = xsum;
		pt.y = ysum;
		pt.z = zsum;
		return true;
	}
	else return false;
}

bool Tracker::search_lost(const std::vector<double> param, const Mat& depthimg, const Mat& hsv, Rect rect, Point3d pt0, Point3d& pt1)
{
	return true;
}

void Tracker::get_clusters(const std::vector<double> param, const Mat& depthimg, const Mat& hsv,
	Point3d pt, std::vector<Cluster>& clusters)
{
	Rect rect;
	radius = 2 * radius;
	calBondingRect(param, pt, rect);
	radius = radius / 2;
	Mat visit = Mat::zeros(depthimg.size(), CV_32S);
	Mat tmp = Mat::zeros(depthimg.size(), CV_8UC3);
	std::queue<Point2i> que;
	int neix[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };
	int neiy[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };
	RNG rng;
	double x, y, z, x1, y1, z1;
	for (int i = 0; i < depthimg.rows; i++)
	for (int j = 0; j < depthimg.cols; j++)
	{
		if (i >= 0 && i < depthimg.rows&&j >= 0 && j < depthimg.cols)
		{
			double d = depthimg.at<double>(i, j);
			x = depth2worldx(param[0], param[2], j, d);
			y = depth2worldy(param[1], param[3], i, d);
			z = d;
			double dist = (pt.x - x)*(pt.x - x) + (pt.y - y)*(pt.y - y) + (pt.z - z)*(pt.z - z);
			dist = sqrt(dist);
			/*if (dist < 2 * radius)
			{
				Vec<uchar, 3> color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				tmp.at<Vec<uchar, 3>>(i, j) = color;
			}*/
			Cluster cluster;
			cluster.pt_num = 1;
			cluster.center.x = x;
			cluster.center.y = y;
			cluster.center.z = z;
			if (d != 0 && visit.at<int>(i, j) == 0/* && dist<2*radius*/)
			{
				Vec<uchar, 3> color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				tmp.at<Vec<uchar, 3>>(i, j) = color;
				visit.at<int>(i, j) = 1;
				que.push(Point2i(i, j));
				while (!que.empty())
				{
					Point2i pt0 = que.front();
					que.pop();
					double d0 = depthimg.at<double>(pt0.x, pt0.y);
					for (int k = 0; k < 8; k++)
					{
						int i1 = pt0.x + neix[k];
						int j1 = pt0.y + neiy[k];
						if (i1 >= 0 && i1 < depthimg.rows&&j1 >= 0 && j1 < depthimg.cols)
						{
							double d1 = depthimg.at<double>(i1, j1);
							x1 = depth2worldx(param[0], param[2], j1, d1);
							y1 = depth2worldx(param[1], param[3], i1, d1);
							z1 = d1;
							double dist1 = (pt.x - x1)*(pt.x - x1) + (pt.y - y1)*(pt.y - y1) + (pt.z - z1)*(pt.z - z1);
							dist1 = sqrt(dist1);
							double dist2 = sqrt((x1 - x)*(x1 - x) + (y1 - y)*(y1 - y) + (z1 - z)*(z1 - z));
							if (d1 != 0 && visit.at<int>(i1, j1) == 0 && fabs(d1 - d0)<100 /*&& dist1<2*radius*/&&dist2<2*radius)
							{
								cluster.pt_num ++;
								cluster.center.x += x;
								cluster.center.y += y;
								cluster.center.z += z;
								visit.at<int>(i1, j1) = 1;
								que.push(Point2i(i1, j1));
								tmp.at<Vec<uchar, 3>>(i1, j1) = color;
							}
						}
					}
				}
			}
			if (cluster.pt_num>200)
			clusters.push_back(cluster);
		}
	}
	for (int i = 0; i < clusters.size(); i++)
	{
		clusters[i].center.x = clusters[i].center.x / clusters[i].pt_num;
		clusters[i].center.y = clusters[i].center.y / clusters[i].pt_num;
		clusters[i].center.z = clusters[i].center.z / clusters[i].pt_num;
		Mat hist1;
		bool a = ms3d(param, depthimg, hsv, target_hist, hist1, clusters[i].center, clusters[i].center);
		clusters[i].score = bhattaCoeff(hist1, target_hist);
		//ms3d(param,depthimg,hsv,)
	}
	imshow("tmp",tmp);
}

void Tracker::initialize(const std::vector<double> param,const Mat& depthimg, const Mat& bgrimg, 
				int h_bins, int s_bins, int v_bins, Rect rect)
{
	hbins = h_bins;
	sbins = s_bins;
	vbins = v_bins;
	//CoordinateConverter cc;
	Mat hsv(bgrimg.size(),CV_8U);
	cvtColor(bgrimg,hsv,CV_BGR2HSV);//bgr to hsv
	Mat gray;
	cvtColor(bgrimg,gray,CV_BGR2GRAY);
	Mat depth = depthimg.clone();

	//medianBlur(depthimg,depth,5);//median filter
	double xsum = 0;
	double ysum = 0;
	double zsum = 0;//center position
	double dsum = 0;
	std::vector<double> zall;
	//std::vector<myPoint> ref_pts;//all the points
	double zs = 0;
	for(int i = rect.y; i < rect.y+rect.height; i++)
		for(int j = rect.x; j < rect.x+rect.width; j++)
		{
			double d = depthimg.at<double>(i,j);
			//std::cout<<d<<std::endl;
			if(d!=0)
			{	
				zall.push_back(d);
				zs += d;
			}
		}
	zs = zs/zall.size();
	sort(zall.begin(),zall.end());
	int mid = zall.size()/2;
	dsum = (zall[mid]);
	double d0 = dsum;
	init_rect = rect;
	depthToWorld(param,rect.x+rect.width/2,rect.y+rect.height/2,dsum,xsum,ysum,zsum);
	double x1,y1,z1;
	depthToWorld(param,rect.x+rect.width/2,rect.y,dsum,x1,y1,z1);
	radius = (x1-xsum)*(x1-xsum)+(y1-ysum)*(y1-ysum)+(z1-zsum)*(z1-zsum);
	radius = sqrt(radius);	
	depthToWorld(param,rect.x+rect.width,rect.y+rect.height,dsum,x1,y1,z1);
	double radius1 = (x1-xsum)*(x1-xsum)+(y1-ysum)*(y1-ysum)+(z1-zsum)*(z1-zsum);
	radius1 = sqrt(radius1);
	if(radius1>radius) radius = radius1;
	if(rect.width/double(rect.height)<1.5&&rect.width/double(rect.height)>0.5)
	{	
		radius = 1.2*radius;
	}
	bandwidth = radius/4;
	target_hist = Mat::zeros(1,hbins*sbins*vbins,CV_64F);
	feature_weight = Mat::ones(1,hbins*sbins*vbins,CV_64F);
	double sum = 0;
	Mat img1 = bgrimg.clone();
	for (int i = rect.x; i < rect.x+rect.width; i++)
		for(int j = rect.y; j < rect.y+rect.height; j++)
		{
			if(i>=0&&i<depthimg.cols&&j>=0&&j<depthimg.rows)
			{
				myPoint pt;
				double d = depthimg.at<double>(j,i);
				depthToWorld(param,i,j,d,pt.x,pt.y,pt.z);
				double dist = (pt.x-xsum)*(pt.x-xsum)+(pt.y-ysum)*(pt.y-ysum)+(pt.z-zsum)*(pt.z-zsum);
				dist = sqrt(dist);
				if(dist<radius&&d>0&&d-d0<300)
				{

					Vec<uchar,3> tmp = hsv.at<Vec<uchar,3>>(j,i);
					pt.h = tmp[0];
					pt.s = tmp[1];
					pt.v = tmp[2];
					//ref_pts.push_back(pt);
					double w = gaussKernel(dist/(bandwidth*bandwidth));
					//double w = dist;
					sum += w;
					int hind = hbins*pt.h/256;
					int sind = sbins*pt.s/256;
					int vind = vbins*pt.v/256;
					int ind = vind*sbins*hbins+sind*hbins+hind;
					target_hist.at<double>(0,ind) += w;
				}
			}
		}
	target_hist = target_hist/sum;
	Point3d cur;
	cur.x = xsum;
	cur.y = ysum;
	cur.z = zsum;
	pts3.push_back(cur);
	isocc = false;
	islost = false;
}

void Tracker::computeContext(const std::vector<double> param, const Mat& depthimg,
	const Mat& hsv, Point3d pt)
{
	feature_weight = Mat::zeros(1, vbins*sbins*hbins, CV_64F);
	double sum = 0;
	int occ_sum = 0;
	int ccn = 0;
	int tcn = 0;
	double context_ratio = 1.6;
	radius = context_ratio*radius;
	Rect br;
	calBondingRect(param, pt, br);
	radius = radius / context_ratio;
	for (int i = br.x; i<br.x + br.width; i = i + 1)
	for (int j = br.y; j<br.y + br.height; j = j + 1)
	{
		if (j >= 0 && j<depthimg.rows&&i >= 0 && i<depthimg.cols)
		{
			double d = depthimg.at<double>(j, i);
			myPoint pt1;
			pt1.x = depth2worldx(param[0], param[2], i, d);
			pt1.y = depth2worldx(param[1], param[3], j, d);
			pt1.z = d;
			//depthToWorld(param,i,j,d,pt1.x,pt1.y,pt1.z);
			double dist = (pt1.x - pt.x)*(pt1.x - pt.x) + (pt1.y - pt.y)*(pt1.y - pt.y) + (pt1.z - pt.z)*(pt1.z - pt.z);
			dist = sqrt(dist);
			if (dist <= radius) tcn++;
			if (dist<context_ratio*radius&&dist>radius)
			{
				ccn ++;
				Vec<uchar, 3> tmp = hsv.at<Vec<uchar, 3>>(j, i);
				double h = tmp[0];
				double s = tmp[1];
				double v = tmp[2];
				int hind = hbins*h / 256;
				int sind = sbins*s / 256;
				int vind = vbins*v / 256;
				pt1.h = h;
				pt1.s = s;
				pt1.v = v;
				int ind = vind*sbins*hbins + sind*hbins + hind;
				feature_weight.at<double>(0, ind) += 1;
				sum += 1;
			}
		}
	}
	con_cn.push_back(ccn);
	tar_cn.push_back(tcn);
	if (con_cn.size() == 1) crc.push_back(0);
	else
	{
		double tmp = ccn - con_cn[con_cn.size() - 2];
		tmp = fabs(tmp) / con_cn[con_cn.size() - 2];
		crc.push_back(tmp);
	}
	if (tar_cn.size() == 1) crt.push_back(0);
	else
	{
		double tmp = tcn - tar_cn[tar_cn.size() - 2];
		tmp = fabs(tmp) / tar_cn[tar_cn.size() - 2];
		crt.push_back(tmp);
	}
	//std::cout << crt[crt.size() - 1] << " " << crc[crc.size() - 1] << std::endl;
	if (sum>500)
	{
		feature_weight = feature_weight / sum;
		for (int i = 0; i < feature_weight.cols; i++)
		{
			double a = feature_weight.at<double>(0, i);
			double b = target_hist.at<double>(0, i);
			if (a<0.00001) a = 0.0001;
			if (b<0.00001) b = 0.0001;
			feature_weight.at<double>(0, i) = (b / a);
		}
	}
	else feature_weight = 1;
}

void Tracker::weightFeatures(const std::vector<double> param,const Mat& depthimg, 
				const Mat& hsv, Rect br, Point3d pt)
{
	feature_weight = Mat::zeros(1,vbins*sbins*hbins,CV_64F);
	double sum = 0;
	int occ_sum = 0;
	for(int i = br.x; i<br.x+br.width; i=i+1)
		for(int j = br.y; j<br.y+br.height; j=j+1)
		{
			if(j>=0&&j<depthimg.rows&&i>=0&&i<depthimg.cols)
			{
				double d = depthimg.at<double>(j,i);
				myPoint pt1;
				pt1.x = depth2worldx(param[0], param[2], i, d);
				pt1.y = depth2worldx(param[1], param[3], j, d);
				pt1.z = d;
				//depthToWorld(param,i,j,d,pt1.x,pt1.y,pt1.z);
				double dist = (pt1.x-pt.x)*(pt1.x-pt.x)+(pt1.y-pt.y)*(pt1.y-pt.y)+(pt1.z-pt.z)*(pt1.z-pt.z);
				if(sqrt(dist)<1.6*radius&&sqrt(dist)>radius)
				{
					Vec<uchar,3> tmp = hsv.at<Vec<uchar,3>>(j,i);
					double h = tmp[0];
					double s = tmp[1];
					double v = tmp[2];
					int hind = hbins*h/256;
					int sind = sbins*s/256;
					int vind = vbins*v/256;
					pt1.h = h;
					pt1.s = s;
					pt1.v = v;
					int ind = vind*sbins*hbins+sind*hbins+hind;
					//dist = sqrt(dist);
					//double w = gaussKernel(dist/(bandwidth*bandwidth));
					feature_weight.at<double>(0,ind) += 1;
					sum += 1;
					if(d<pt.z) occ_sum ++;
				}
			}	
		}
		if(sum>500)
		{
			feature_weight = feature_weight/sum;
			for( int i = 0; i < feature_weight.cols; i++)
			{
				double a = feature_weight.at<double>(0,i);
				double b = target_hist.at<double>(0,i);
				if( a<0.00001 ) a = 0.0001;
				if( b<0.00001 ) b = 0.0001;
				feature_weight.at<double>(0,i) = (b/a);
			}
		}
		else feature_weight = 1;
}

void Tracker::track_all()
{
	double total_time = 0;
	double total_frame = 0;
	Rect selection;
	Mat image;
	bool istrackerinit = false;
	bool is_contiune = true;
	std::ofstream outfile;
	outfile.open("time.txt");
	std::ifstream infile;
	infile.open("param.txt");
	std::vector<double> param;
	double a;
	for (int i = 0; i < 4; i++)
	{
		infile >> a;
		param.push_back(a);
	}
	int k = 0;
	char rgbname[500];
	char depthname[500];
	sprintf(rgbname, "rgb//%d.png", k);
	sprintf(depthname, "depth//%d.png", k);

	Mat imgrgb = imread(rgbname);
	Mat imgdepth = imread(depthname, CV_LOAD_IMAGE_ANYDEPTH);
	cv::Mat hsv, depthimg, depthimg_pre, hsv_pre;
	while (imgrgb.rows>0)
	{
		cvtColor(imgrgb, hsv, CV_BGR2HSV);
		image = imgrgb.clone();
		if (!istrackerinit)
		{
			rectangle(image, selection, Scalar(0, 0, 255));
			char c1;
			std::ifstream infile1;
			infile1.open("init.txt");
			infile1 >> selection.x;
			infile >> c1;
			infile1 >> selection.y;
			infile >> c1;
			infile1 >> selection.width;
			infile >> c1;
			infile1 >> selection.height;
			infile1.close();
			imgdepth.convertTo(depthimg, CV_64F);
			depthimg = depthimg / 8;
			initialize(param, depthimg, imgrgb, 10, 10, 10, selection);
			computeContext(param, depthimg, hsv, pts3[pts3.size() - 1]);
			depthimg_pre = depthimg.clone();
			hsv_pre = imgrgb.clone();
			istrackerinit = true;
		}
		if (istrackerinit&&is_contiune)
		{
			Mat dst;
			imgdepth.convertTo(depthimg, CV_64F);
			depthimg = depthimg / 8;
			Rect br3;
			Point3d ptm = get_curpos();
			calBondingRect(param, ptm, br3);
			clock_t start, finish;
			start = clock();
			//track_with_detector(param, depthimg, hsv, depthimg_pre, hsv_pre);
			track(param, depthimg, hsv, depthimg_pre, hsv_pre);
			//track_ms3d(param, depthimg, hsv, depthimg_pre, hsv_pre);
			if (k % 3 != 0||isocc) feature_weight = 1;
			finish = clock();
			double duration = (double)(finish - start) / CLOCKS_PER_SEC;
			total_time += duration;
			total_frame++;
			std::cout << duration << "seconds" << std::endl;
			
			Point3d pt1 = get_curpos();
			Rect br2;
			br2 = get_cur_rect();
			Scalar color;
			if (is_occluded()) color = Scalar(0, 0, 255);
			else color = Scalar(0, 255, 0);
			showTarget(param, image, depthimg, br2, pt1); 
			rectangle(image, br2, color, 3);
			char name[500];
			sprintf(name, "result//%d.png", k);
			imshow("Color Image", image);
			//imshow("Depth", depthimg/8000);
			Rect rect1;
			rect1.x = br2.x - br2.width / 2;
			rect1.y = br2.y - br2.height / 2;
			rect1.height = br2.height * 2;
			rect1.width = br2.width * 2;
			//get_clusters(param, depthimg, rect1, pt1);
			char c = (char)waitKey(1);
			k++;
			depthimg_pre = depthimg.clone();
			hsv_pre = hsv.clone();
		}
		sprintf(rgbname, "rgb//%d.png", k);
		sprintf(depthname, "depth//%d.png", k);
		imgrgb = imread(rgbname);
		imgdepth = imread(depthname, CV_LOAD_IMAGE_ANYDEPTH);
		image = imgrgb.clone();
	}
	saveTrajBox("box.txt");
	outfile << total_time << ' ' << total_frame << std::endl;
	outfile.close();
}

void Tracker::track_ms3d(const std::vector<double> param, const Mat& depthimg, const Mat& hsv,
	const Mat& depthimg_pre, const Mat& hsv_pre)
{
	Point3d pt0 = get_curpos();
	Point3d pt1;
	Mat hist1;
	bool a = ms3d(param, depthimg, hsv, target_hist, hist1, pt0, pt1);
	isocc = false;
	status.push_back(1);
	Rect br1;
	calBondingRect1(param, pt1, br1);
	rects.push_back(br1);
	pts3.push_back(pt1);
	computeContext(param, depthimg, hsv, pts3[pts3.size() - 1]);
	if (!isocc) tracker_adaption(hist1);
}


void Tracker::tracker_adaption(const Mat& hist1)
{
	if (crc.size() > 1)
	{
		if (target_hist.cols==hist1.cols&&
			crc[crc.size() - 1] < 0.15&&crc[crc.size() - 2] < 0.15&&
			crt[crt.size() - 1] < 0.15&&crt[crt.size() - 2] < 0.15)
		{
			target_hist = target_hist + 0.1*hist1;
			double sum = 0;
			for (int i = 0; i < target_hist.cols; i++) sum += target_hist.at<double>(0, i);
			target_hist = target_hist / sum;
		}
	}
}

void Tracker::track_with_detector(const std::vector<double> param, const Mat& depthimg, const Mat& hsv,
	const Mat& depthimg_pre, const Mat& hsv_pre)
{
	Point3d pt0 = get_curpos();
	Point3d pt1;
	Mat hist1;
	bool a = ms3d(param, depthimg, hsv, target_hist, hist1, pt0, pt1);

	//iter = iternum;
	islost = false;
	if (!isocc)
	{
		Mat hist11;
		Point3d pt11;
		bool a1 = ms3d(param, depthimg_pre, hsv_pre, target_hist, hist11, pt1, pt11);
		double depth_diff = pt11.z - pt0.z;
		bool is_distracted = false;
		if ((-depth_diff)>radius / 1.3) is_distracted = true;
		Rect br1;
		if (is_distracted || (!a))
		{
			if (is_distracted) occ_pt = pt1;

			pt1 = pt0;
			br1 = rects[rects.size() - 1];
			isocc = true;
			if (!a)
			{
				bool init_success = get_rect_pos3(param, depthimg, rects[rects.size() - 1], occ_pt);
				if ((!init_success))
				{
					isocc = false;
					islost = true;
				}
			}
			status.push_back(0);
		}
		else
		{
			isocc = false;
			status.push_back(1);
			calBondingRect1(param, pt1, br1);
		}
		//scores.push_back(fscore);
		rects.push_back(br1);
		pts3.push_back(pt1);
	}
	else
	{
		//bool init_success = get_rect_pos3(param, depthimg, rects[rects.size() - 1], occ_pt);

		Mat hist11;
		Point3d pt11;
		bool a1 = ms3d(param, depthimg_pre, hsv_pre, target_hist, hist11, occ_pt, pt11);
		double diff1 = fabs(pt1.z - pt0.z);
		double diff2 = fabs(pt1.z - pt11.z);
		Rect br1;
		if (diff1 < diff2&&a)//
		{
			isocc = false;
			status.push_back(1);
			calBondingRect1(param, pt1, br1);
		}
		else
		{
			pt1 = pt0;
			isocc = true;
			br1 = rects[rects.size() - 1];
			status.push_back(0);
			occ_pt = pt11;
		}
		rects.push_back(br1);
	}
	computeContext(param, depthimg, hsv, pts3[pts3.size() - 1]);
	if (isocc)
	{
		occ_cn++;
		std::vector<Cluster> clusters;
		get_clusters(param, depthimg, hsv, pts3[pts3.size() - 1], clusters);
		double max_score = 0;
		int max_id = -1;
		for (int i = 0; i < clusters.size(); i++)
		{
			if (clusters[i].score>max_score)
			{
				max_score = clusters[i].score;
				max_id = i;
			}
		}
		if (max_id>-1) det_clusters.push_back(clusters[max_id]);
		int n = det_clusters.size();
		if (n > 2)
		{
			double dist1 = ptDist(det_clusters[n - 1].center, det_clusters[n - 2].center);
			double dist2 = ptDist(det_clusters[n - 2].center, det_clusters[n - 3].center);
			if (dist1<radius / 4 && dist2<radius/4 && det_clusters[n - 1].score>0.2
				&&det_clusters[n - 2].score > 0.2&&det_clusters[n - 3].score > 0.2)
			{
				isocc = false;
				status.push_back(1);
				pt1 = det_clusters[n - 1].center;
				Rect br1;
				calBondingRect1(param, pt1, br1);
				rects[rects.size()-1] = br1;
				pts3[pts3.size()-1] = pt1;
			}
		}
	}
	else
	{
		occ_cn = 0;
		det_clusters.clear();
	}
}

void Tracker::track(const std::vector<double> param,const Mat& depthimg, const Mat& hsv,
	const Mat& depthimg_pre, const Mat& hsv_pre)
{
	Point3d pt0 = get_curpos();
	Point3d pt1;
	Mat hist1;
	if (!is_occluded() && pts3.size() % 3 == 0&&!islost)
	{
		Rect br;
		radius = 2 * radius;
		calBondingRect(param, pt0, br);
		radius = 0.5*radius;
		weightFeatures(param, depthimg, hsv, br, pt0);
	}
	//meanshift3d(param,depthimg,hsv,pt0,pt1);
 	bool a = ms3d(param, depthimg, hsv, target_hist, hist1, pt0, pt1);
	
	//iter = iternum;
	islost = false;
	if (!isocc)
	{
		Mat hist11;
		Point3d pt11;
		bool a1 = ms3d(param, depthimg_pre, hsv_pre, target_hist, hist11, pt1, pt11);//backward tracking
		double depth_diff = pt11.z - pt0.z;
		bool is_distracted = false;
		if ((-depth_diff)>radius / 1.3) is_distracted = true;
		Rect br1;
		if (is_distracted || (!a))
		{
			if (is_distracted) occ_pt = pt1;
			
			pt1 = pt0;
			br1 = rects[rects.size() - 1];
			isocc = true;
			if (!a)
			{
				bool init_success = get_rect_pos3(param, depthimg, rects[rects.size() - 1], occ_pt);
				if ((!init_success))
				{
					isocc = false;
					islost = true;
				}
			}
			status.push_back(0);
		}
		else
		{
			isocc = false;
			status.push_back(1);
			calBondingRect1(param, pt1, br1);
		}
		//scores.push_back(fscore);
		rects.push_back(br1);
		pts3.push_back(pt1);
	}
	else
	{
		//bool init_success = get_rect_pos3(param, depthimg, rects[rects.size() - 1], occ_pt);
		
		Mat hist11;
		Point3d pt11;
		bool a1 = ms3d(param, depthimg_pre, hsv_pre, target_hist, hist11, occ_pt, pt11);
		double diff1 = fabs(pt1.z - pt0.z);
		double diff2 = fabs(pt1.z - pt11.z);
		Rect br1;
		if (diff1 < diff2&&a)//OTR
		{
			isocc = false;
			status.push_back(1);
			calBondingRect1(param, pt1, br1);
			//std::cout << "OTR: " << diff2 / diff1 << std::endl;
		}
		else
		{
			pt1 = pt0;
			isocc = true;
			br1 = rects[rects.size() - 1];
			status.push_back(0);
			occ_pt = pt11;
		}
		rects.push_back(br1);
	}
	computeContext(param, depthimg, hsv, pts3[pts3.size() - 1]);
	if (!isocc) tracker_adaption(hist1);
	//if (isocc) occ_cn++;
	//else occ_cn = 0;
	//std::cout << "occ_cn: " << occ_cn << std::endl;
}

void Tracker::cal_hist(const std::vector<double> param,const Mat& hsv, const Mat& depthimg, Mat& hist, Rect br, 
		std::vector<int>& ref, std::vector<myPoint>& pts, Point3d pt, Point3d& center, int& pt_cn)
{
	if (hist.cols!=vbins*sbins*hbins)
		hist = Mat::zeros(1,vbins*sbins*hbins,CV_64F);
	center.x = 0;
	center.y = 0;
	center.z = 0;
	double sum = 0;
	pt_cn = 0;
	if (ref.size()<br.width*br.height)
	{
		ref.resize(br.width*br.height);
		pts.resize(br.width*br.height);
	}
	for(int i = br.x; i<br.x+br.width; i++)
		for(int j = br.y; j<br.y+br.height; j++)
		{
			if(j>=0&&j<depthimg.rows&&i>=0&&i<depthimg.cols)
			{
				double d = depthimg.at<double>(j,i);
				myPoint pt1;
				pt1.x = depth2worldx(param[0], param[2], i, d);
				pt1.y = depth2worldx(param[1], param[3], j, d);
				pt1.z = d;
				//depthToWorld(param,i,j,d,pt1.x,pt1.y,pt1.z);
				double dist = (pt1.x-pt.x)*(pt1.x-pt.x)+(pt1.y-pt.y)*(pt1.y-pt.y)+(pt1.z-pt.z)*(pt1.z-pt.z);
				if(sqrt(dist)<radius)
				{
					Vec<uchar,3> tmp = hsv.at<Vec<uchar,3>>(j,i);
					double h = tmp[0];
					double s = tmp[1];
					double v = tmp[2];
					int hind = hbins*h/256;
					int sind = sbins*s/256;
					int vind = vbins*v/256;
					pt1.h = h;
					pt1.s = s;
					pt1.v = v;
					int ind = vind*sbins*hbins+sind*hbins+hind;
					double w = gaussKernel(dist/(bandwidth*bandwidth));
					hist.at<double>(0,ind) += w;
					sum += w;
					pts[pt_cn] = pt1;
					ref[pt_cn] = ind;
					center.x += pt1.x;
					center.y += pt1.y;
					center.z += pt1.z;
					pt_cn ++;
				}
			}	
		}
	hist = hist/sum;
	center.x = center.x/pts.size();
	center.y = center.y/pts.size();
	center.z = center.z/pts.size();
}


void Tracker::cal_hist(const std::vector<double> param,const Mat& hsv, const Mat& depthimg, Mat& hist, 
			Rect br, Point3d pt)
{
	hist = Mat::zeros(1,vbins*sbins*hbins,CV_64F);
	double sum = 0;
	for(int i = br.x; i<br.x+br.width; i++)
		for(int j = br.y; j<br.y+br.height; j++)
		{
			if(j>=0&&j<depthimg.rows&&i>=0&&i<depthimg.cols)
			{
				double d = depthimg.at<double>(j,i);
				myPoint pt1;
				pt1.x = depth2worldx(param[0], param[2], i, d);
				pt1.y = depth2worldx(param[1], param[3], j, d);
				pt1.z = d;
				//depthToWorld(param,i,j,d,pt1.x,pt1.y,pt1.z);
				double dist = (pt1.x-pt.x)*(pt1.x-pt.x)+(pt1.y-pt.y)*(pt1.y-pt.y)+(pt1.z-pt.z)*(pt1.z-pt.z);
				if(sqrt(dist)<radius)
				{
					Vec<uchar,3> tmp = hsv.at<Vec<uchar,3>>(j,i);
					double h = tmp[0];
					double s = tmp[1];
					double v = tmp[2];
					int hind = hbins*h/256;
					int sind = sbins*s/256;
					int vind = vbins*v/256;
					pt1.h = h;
					pt1.s = s;
					pt1.v = v;
					int ind = vind*sbins*hbins+sind*hbins+hind;
					//dist = sqrt(dist);
					double w = gaussKernel(dist/(bandwidth*bandwidth));
					hist.at<double>(0,ind) += w;
					sum += w;
				}
			}	
		}
	hist = hist/sum;
}

void Tracker::showPdf2d(const Mat& hsv)
{
	int width = init_rect.width;
	int height = init_rect.height;
	Rect rect;
	rect.height = height;
	rect.width = width;
	double bw = double(width) / 3;
	Mat index_map(hsv.rows, hsv.cols, CV_32S);
	Mat pdf_map = Mat::zeros(hsv.rows, hsv.cols, CV_64F);
	Mat pdf_show = Mat::zeros(hsv.rows, hsv.cols, CV_8U);
	for (int x = 0; x < hsv.cols; x++)
	{
		for (int y = 0; y < hsv.rows; y++)
		{
			Vec<uchar, 3> tmp = hsv.at<Vec<uchar, 3>>(y, x);
			double h = tmp[0];
			double s = tmp[1];
			double v = tmp[2];
			int hind = hbins*h / 256;
			int sind = sbins*s / 256;
			int vind = vbins*v / 256;
			int ind = vind*sbins*hbins + sind*hbins + hind;
			index_map.at<int>(y, x) = ind;
		}
	}
	double max_value = 0;
	for (int x = -width / 2; x < hsv.cols - width/2; x++)
	{
		std::cout << x << ' ' << std::endl;
		for (int y = -height / 2; y < hsv.rows - height/2; y++)
		{

			rect.x = x;
			rect.y = y;
			int x2 = x + width / 2;
			int y2 = y + height / 2;
			Mat hist1 = Mat::zeros(1, vbins*sbins*hbins, CV_64F);
			double wsum = 0;
			for (int x1 = rect.x; x1<rect.x + rect.width; x1++)
			for (int y1 = rect.y; y1<rect.y + rect.height; y1++)
			{
				if (x1 >= 0 && x1 < hsv.cols&&y1 >= 0 && y1 < hsv.rows)
				{
					Vec<uchar, 3> tmp = hsv.at<Vec<uchar, 3>>(y1, x1);
					double h = tmp[0];
					double s = tmp[1];
					double v = tmp[2];
					int hind = hbins*h / 256;
					int sind = sbins*s / 256;
					int vind = vbins*v / 256;
					int ind = vind*sbins*hbins + sind*hbins + hind;
					double weight = (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2);
					weight = weight / (bw*bw);
					weight = gaussKernel(weight);
					hist1.at<double>(0, ind) += weight;
					wsum += weight;
				}	
			}
			hist1 = hist1 / wsum;
			double psum = 0;
			for (int x1 = rect.x; x1<rect.x + rect.width; x1++)
			for (int y1 = rect.y; y1<rect.y + rect.height; y1++)
			{
				if (x1 >= 0 && x1 < hsv.cols&&y1 >= 0 && y1 < hsv.rows)
				{
					double weight = (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2);
					weight = weight / (bw*bw);
					weight = gaussKernel(weight);
					int ind = index_map.at<int>(y1, x1);
					double tmp = target_hist.at<double>(0, ind) / hist1.at<double>(0, ind);
					//double tmp = target_hist.at<double>(0, ind);
					psum += weight*sqrt(tmp);
				}
			}
			pdf_map.at<double>(y2, x2) = psum;
			if (psum>max_value) max_value = psum;
		}
	}
	for (int i = 0; i < pdf_show.rows; i++)
	for (int j = 0; j < pdf_show.cols; j++)
	{
		int a = 255.0*(pdf_map.at<double>(i, j) / max_value);
		pdf_show.at<uchar>(i, j) = a;
	}
	saveMatrix("pdf.txt", pdf_map);
	//imshow("pdf", pdf_show);
	//waitKey();
}

bool Tracker::ms3d(const std::vector<double> param, const Mat& depthimg,
	const Mat& hsv, const Mat& hist0, Mat& hist1, Point3d pt0, Point3d& pt1)
{
	double dist_thre = 5;
	bool flag = true; //flag that indicates if sufficient number of points are bounded
	Point3d pt; //the starting point of outer loop
	pt = pt0;
	Mat hist;
	double coeff;
	int out_iter = 0;
	double fscore;
	int pt_num;
	std::vector<int> ref1(320 * 240, 0);
	myPoint ptt;
	std::vector<myPoint> pts1(320 * 240, ptt);
	while (flag&&out_iter<5)
	{
		Point3d cen;
		Rect br;
		calBondingRect(param, pt, br);
		cal_hist(param, hsv, depthimg, hist, br, ref1, pts1, pt, cen, pt_num);
		double coef = bhattaCoeff(hist0, hist);
		if (pt_num>100)// caused by tracker lost (fast motion) or occlusion
		{
			double tmpx1 = 0;
			double tmpy1 = 0;
			double tmpz1 = 0;
			double tmp2 = 0;
			int k = 0;
			for (int i = 0; i < pt_num; i++)
			{
				int bind = ref1[i];
				double w = sqrt(feature_weight.at<double>(0, bind)*hist0.at<double>(0, bind) / hist.at<double>(0, bind));
				double x = pts1[i].x;
				double y = pts1[i].y;
				double z = pts1[i].z;
				double dist = (x - pt.x)*(x - pt.x) + (y - pt.y)*(y - pt.y) + (z - pt.z)*(z - pt.z);
				double g = gaussKernelDev(dist / (bandwidth*bandwidth));
				tmpx1 += w*g*x;
				tmpy1 += w*g*y;
				tmpz1 += w*g*z;
				tmp2 += w*g;
			}
			pt1.x = tmpx1 / tmp2;
			pt1.y = tmpy1 / tmp2;
			pt1.z = tmpz1 / tmp2;

			Point3d cen1;
			Rect br1;
			calBondingRect(param, pt1, br1);
			cal_hist(param, hsv, depthimg, hist1, br1, pt1);
			double coef1 = bhattaCoeff(target_hist, hist1);
			//std::cout<<coef2<<std::endl;
			int inner_it = 0;
			double dist1 = (pt.x - pt1.x)*(pt.x - pt1.x) + (pt.y - pt1.y)*(pt.y - pt1.y) + (pt.z - pt1.z)*(pt.z - pt1.z);
			dist1 = sqrt(dist1);
			while (coef1<coef&&inner_it<2 && dist1>dist_thre)
			{
				dist1 = dist1 / 2;
				pt1.x = (pt.x + pt1.x) / 2;
				pt1.y = (pt.y + pt1.y) / 2;
				pt1.z = (pt.z + pt1.z) / 2;
				calBondingRect(param, pt1, br1);
				cal_hist(param, hsv, depthimg, hist1, br1, pt1);
				coef1 = bhattaCoeff(target_hist, hist1);
				inner_it++;
			}
			if (coef1>coef)
			{
				pt.x = pt1.x;
				pt.y = pt1.y;
				pt.z = pt1.z;
			}
			else
			{
				pt1 = pt;
				hist1 = hist.clone();
				flag = false;
			}
		}
		else{ flag = false; return false; }
		out_iter++;
	}
	return true;
}

void Tracker::calBondingRect1(const std::vector<double> param,Point3d pt, Rect& br)
{
	double x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4;
	double xd1 = init_rect.x+init_rect.width/2;
	double yd1 = init_rect.y+init_rect.height/2;
	double d0 = pts3[0].z;
	depthToWorld(param,xd1,yd1,d0,x1,y1,z1);
	depthToWorld(param,xd1+5,yd1,d0,x2,y2,z2);
	double dist1 = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2);
	dist1 = sqrt(dist1);
	double xd2,yd2,d2;
	worldToDepth(param,xd2,yd2,d2,pt.x,pt.y,pt.z);
	depthToWorld(param,xd2+5,yd2,d2,x4,y4,z4);
	double dist2 = (pt.x-x4)*(pt.x-x4)+(pt.y-y4)*(pt.y-y4)+(pt.z-z4)*(pt.z-z4);
	dist2 = sqrt(dist2);	
	br.width = init_rect.width*dist1/dist2;
	br.height = init_rect.height*dist1/dist2;
	br.x = xd2-0.5*br.width;
	br.y = yd2-0.5*br.height;
}

void Tracker::calBondingRect(const std::vector<double> param,Point3d pt, Rect& br)
{
	int vnum = 8;
	double dx[8] = {-radius, radius, radius, -radius,-radius, radius, radius, -radius};
	double dy[8] = {-radius, -radius, radius, radius,-radius, -radius, radius, radius};
	double dz[8] = {radius, radius, radius, radius,-radius, -radius, -radius, -radius};
	double scale = 1;
	double minx = 10000000;
	double miny = 10000000;
	double maxx = -10000000;
	double maxy = -10000000;
	for(int i = 0; i < vnum; i++)
	{
		double x = pt.x+dx[i]*scale;
		double y = pt.y+dy[i]*scale;
		double z = pt.z+dz[i]*scale;
		double depthx;
		double depthy;
		double depth;
		worldToDepth(param,depthx,depthy,depth,x,y,z);
		if(depthx<minx) minx = depthx;
		if(depthy<miny) miny = depthy;
		if(depthx>maxx) maxx = depthx;
		if(depthy>maxy) maxy = depthy;
	}
	br.x = minx-3;
	br.y = miny-3;
	br.width = maxx-minx+6;
	br.height = maxy-miny+6;
	if (br.width>600 || br.height>600)
	{
		br.width = init_rect.width;
		br.height = init_rect.height;
	}
}

void Tracker::depthToWorld(const std::vector<double> param,double depthx,double depthy,double depth,double & wx,double & wy,double & wz)
{
	double fx = param[0];
	double fy = param[1];
	double cx = param[2];
	double cy = param[3];
	wz = depth;
	wx = (depthx - cx) * depth / fx;
	wy = (depthy - cy) * depth / fy;
}

void Tracker::worldToDepth(const std::vector<double> param,double& depthx,double& depthy,double& depth,double wx,double wy,double wz)
{
	double fx = param[0];
	double fy = param[1];
	double cx = param[2];
	double cy = param[3];
	depthx = (fx*wx+cx*wz)/wz;
	depthy = (fy*wy+cy*wz)/wz;
	depth = wz;
}

void Tracker::showTarget(const std::vector<double> param,Mat& img,const Mat& depthimg, Rect& br, Point3d pt)
{
	double mean_x = 0;
	double mean_y = 0;
	double cn = 0;
	if(br.x>-br.width&&br.x<depthimg.cols+br.width&&br.y>-br.height&&br.y<br.height+depthimg.rows)
	{
		for(int i = br.x; i<br.x+br.width; i++)
		for(int j = br.y; j<br.y+br.height; j++)
		{
			if(j>=0&&j<depthimg.rows&&i>=0&&i<depthimg.cols)
			{
				double d = depthimg.at<double>(j,i);
				myPoint pt1;
				depthToWorld(param,i,j,d,pt1.x,pt1.y,pt1.z);
				double dist = (pt1.x-pt.x)*(pt1.x-pt.x)+(pt1.y-pt.y)*(pt1.y-pt.y)+(pt1.z-pt.z)*(pt1.z-pt.z);
				if(sqrt(dist)<radius)
				{
					Vec<uchar,3> tmp = img.at<Vec<uchar,3>>(j,i);
					tmp[0] = (tmp[0]+0)/2;
					tmp[1] = (tmp[1]+255)/2;
					tmp[2] = (tmp[2]+0)/2;
					img.at<Vec<uchar,3>>(j,i) = tmp;
					mean_x += i;
					mean_y += j;
					cn ++;
				}
			}	
		}
		if(cn>200&&!is_occluded())
		{
			mean_x = mean_x/cn;
			mean_y = mean_y/cn;
			br.x = mean_x-br.width/2;
			br.y = mean_y-br.height/2;
			rects[rects.size()-1] = br;
		}
	}
	
}

void Tracker::saveTraj(char* filename)
{
	std::ofstream outfile;
	outfile.open(filename);
	outfile<<pts3.size()-1<<std::endl;
	outfile<<radius<<std::endl;
	for(int i = 1; i < pts3.size(); i++)
	{
		outfile<<pts3[i].x<<" "<<pts3[i].y<<" "<<pts3[i].z<<std::endl;
	}
	outfile.close();
}

void Tracker::saveTrajBox(char* filename)
{
	std::ofstream outfile;
	outfile.open(filename);
	//outfile<<rects.size()<<std::endl;
	//outfile<<radius<<std::endl;
	for(int i = 0; i < rects.size(); i++)
	{
		double x1 = rects[i].x;
		double y1 = rects[i].y;
		double x2 = rects[i].x+rects[i].width;
		double y2 = rects[i].y+rects[i].height;
		if(status[i]==1)
			outfile<<x1<<","<<y1<<","<<x2<<","<<y2<<","<<0<<std::endl;
		else
			outfile<<"NaN"<<","<<"NaN"<<","<<"NaN"<<","<<"NaN"<<","<<1<<std::endl;
	}
	outfile.close();
}


void Tracker::readTraj(std::vector<Point3d>& pts, char* filename, double & radius)
{
	std::ifstream infile;
	infile.open(filename);
	int n;
	infile>>n;
	infile>>radius;
	for(int i = 0; i < n; i++)
	{
		Point3d pt;
		infile>>pt.x;
		infile>>pt.y;
		infile>>pt.z;
	}
	infile.close();
}

Tracker::~Tracker(void)
{
}

double bhattaCoeff(const Mat& hist1, const Mat& hist2)
{
	double tmp = 0;
	for(int i = 0; i < hist1.cols; i++)
	{
		tmp += sqrt(hist1.at<double>(0,i)*hist2.at<double>(0,i));
		//std::cout<<sqrt(hist1.at<double>(0,i)*hist2.at<double>(0,i))<<std::endl;
	}
	return tmp;
}

void saveMatrix(char* filename, const Mat & m)
{
	std::ofstream outfile;
	outfile.open(filename);
		for(int i = 0; i < m.rows; i++)
		{	
			for(int j = 0; j < m.cols; j++)
			{
				double a = m.at<double>(i,j);
				outfile<<a<<" ";
			}
			outfile<<std::endl;
		}
		outfile.close();
}

double ptDist(const Point3d& pt1, const Point3d& pt2)
{
	double dist = (pt1.x - pt2.x)*(pt1.x - pt2.x) + (pt1.y - pt1.y)*(pt1.y - pt1.y) + (pt1.z - pt2.z)*(pt1.z - pt2.z);
	dist = sqrt(dist);
	return dist;
}
